// Written by K. M. Knausgård 2026-02-14, based on https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
//
// C++23, std-only. Max-performance CPU kernels (no autograd graph).
//
// Build in Release for real speed.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <expected>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

// =====================================================================
//  Model constants - all constexpr
// =====================================================================
constexpr int nEmbd     = 16;
constexpr int nHead     = 4;
constexpr int nLayer    = 1;
constexpr int blockSize = 16;

static_assert(nEmbd > 0 && nHead > 0 && nLayer > 0 && blockSize > 0);
static_assert(nEmbd % nHead == 0);

constexpr int headDim    = nEmbd / nHead;
constexpr int mlpHidden  = 4 * nEmbd;

// All hot dimensions are multiples of 4 (required by ILP dot)
static_assert(headDim % 4 == 0);
static_assert(nEmbd % 4 == 0);
static_assert(mlpHidden % 4 == 0);

constexpr float initStd  = 0.08F;
constexpr float rmsEps   = 1e-5F;

constexpr float learningRate = 0.01F;
constexpr float beta1        = 0.85F;
constexpr float beta2        = 0.99F;
constexpr float epsAdam      = 1e-8F;

constexpr int   defaultSteps       = 1000;
constexpr int   defaultSamples     = 20;
constexpr int   defaultLogEvery    = 25;
constexpr float defaultTemperature = 0.5F;

// Pre-computed 1/sqrt(headDim) at compile time via Newton-Raphson
constexpr float invSqrtHeadDim = []() constexpr {
    double x = static_cast<double>(headDim);
    double g = x * 0.5;
    for (int i = 0; i < 20; ++i) g = (g + x / g) * 0.5;
    return static_cast<float>(1.0 / g);
}();

// =====================================================================
//  Errors
// =====================================================================
enum class ErrorCode : uint8_t {
    fileOpenFailed, fileReadFailed, emptyDataset, invalidChar
};
struct Error { ErrorCode code{}; std::string_view message{}; };
template <class T> using Expected = std::expected<T, Error>;

// =====================================================================
//  Utilities
// =====================================================================
inline size_t toSize(int v) noexcept { return static_cast<size_t>(v); }
inline uint32_t toU32(int v) noexcept { return static_cast<uint32_t>(v); }

using Clock = std::chrono::steady_clock;
inline double toMilliseconds(Clock::duration d) noexcept {
    return std::chrono::duration<double, std::milli>{d}.count();
}
inline double toMicroseconds(Clock::duration d) noexcept {
    return std::chrono::duration<double, std::micro>{d}.count();
}

inline int parseInt(std::string_view s, int fallback) noexcept {
    int v{fallback};
    auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc{} || res.ptr != s.data() + s.size()) return fallback;
    return v;
}

inline float parseFloat(std::string_view s, float fallback) noexcept {
    if (s.empty()) return fallback;
    int sign{1}; size_t i{0};
    if (s[0] == '-') { sign = -1; i = 1; } else if (s[0] == '+') { i = 1; }
    float intPart{0.0F}; bool any{false};
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (c >= '0' && c <= '9') { any = true; intPart = intPart * 10.0F + float(c - '0'); }
        else break;
    }
    float fracPart{0.0F}, fracScale{1.0F};
    if (i < s.size() && s[i] == '.') {
        ++i;
        for (; i < s.size(); ++i) {
            char c = s[i];
            if (c >= '0' && c <= '9') {
                any = true; fracScale *= 10.0F;
                fracPart = fracPart * 10.0F + float(c - '0');
            } else return fallback;
        }
    } else if (i != s.size()) return fallback;
    if (!any) return fallback;
    return (intPart + fracPart / fracScale) * float(sign);
}

// =====================================================================
//  Blitz++-style expression templates  (element-wise float vectors)
//
//  Usage:
//    assign<N>(out, vec(a) + vec(b));           // fused a[i]+b[i]
//    assign<N>(out, vec(a) + scale(vec(b), s)); // fused a[i]+b[i]*s
//    accum<N>(out, vec(a));                     // fused out[i]+=a[i]
//
//  The compiler sees a single loop with the full expression inlined.
//  Zero heap temporaries; the expression tree lives in registers.
// =====================================================================
struct VecLeaf {
    const float* __restrict__ p;
    float operator[](int i) const noexcept { return p[i]; }
};
inline VecLeaf vec(const float* p) noexcept { return {p}; }

template <typename L, typename R> struct VecAdd {
    L l; R r;
    float operator[](int i) const noexcept { return l[i] + r[i]; }
};
template <typename L, typename R> struct VecSub {
    L l; R r;
    float operator[](int i) const noexcept { return l[i] - r[i]; }
};
template <typename E> struct VecScale {
    E e; float s;
    float operator[](int i) const noexcept { return e[i] * s; }
};

template <typename L, typename R>
auto operator+(L l, R r) noexcept -> VecAdd<L, R> { return {l, r}; }
template <typename L, typename R>
auto operator-(L l, R r) noexcept -> VecSub<L, R> { return {l, r}; }
template <typename E>
auto scale(E e, float s) noexcept -> VecScale<E> { return {e, s}; }

template <int N, typename Expr>
inline void assign(float* __restrict__ out, Expr expr) noexcept {
    for (int i = 0; i < N; ++i) out[i] = expr[i];
}
template <int N, typename Expr>
inline void accum(float* __restrict__ out, Expr expr) noexcept {
    for (int i = 0; i < N; ++i) out[i] += expr[i];
}

// =====================================================================
//  Dataset / tokenizer  (logic unchanged)
// =====================================================================
Expected<std::vector<std::string>> readDocs(std::string_view path) noexcept {
    std::ifstream in{std::string{path}, std::ios::binary};
    if (!in.is_open())
        return std::unexpected(Error{ErrorCode::fileOpenFailed, "failed to open dataset file"});
    std::vector<std::string> docs; std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) docs.push_back(line);
    }
    if (!in.good() && !in.eof())
        return std::unexpected(Error{ErrorCode::fileReadFailed, "failed while reading dataset file"});
    if (docs.empty())
        return std::unexpected(Error{ErrorCode::emptyDataset, "dataset is empty"});
    return docs;
}

struct Tokenizer {
    std::string uchars{};
    std::array<int, 256> charToId{};
    int bosId{-1};
    int vocabSize{0};

    static Expected<Tokenizer> build(const std::vector<std::string>& docs) noexcept {
        Tokenizer t{}; t.charToId.fill(-1);
        std::array<bool, 256> seen{}; seen.fill(false);
        for (const auto& d : docs) for (unsigned char c : d) seen[c] = true;
        t.uchars.clear(); t.uchars.reserve(256);
        for (int c = 0; c < 256; ++c)
            if (seen[toSize(c)]) t.uchars.push_back(static_cast<char>(c));
        std::sort(t.uchars.begin(), t.uchars.end());
        for (int i = 0; i < static_cast<int>(t.uchars.size()); ++i)
            t.charToId[static_cast<unsigned char>(t.uchars[toSize(i)])] = i;
        t.bosId = static_cast<int>(t.uchars.size());
        t.vocabSize = t.bosId + 1;
        return t;
    }
    Expected<int> encodeChar(char c) const noexcept {
        int id = charToId[static_cast<unsigned char>(c)];
        if (id < 0)
            return std::unexpected(Error{ErrorCode::invalidChar, "encountered char not in vocab"});
        return id;
    }
    char decodeId(int id) const noexcept { return uchars[toSize(id)]; }
};

// =====================================================================
//  Model param layout
// =====================================================================
struct ModelOffsets {
    uint32_t wte{0u}, wpe{0u}, lmHead{0u};
    std::array<uint32_t, nLayer> attnWq{}, attnWk{}, attnWv{}, attnWo{};
    std::array<uint32_t, nLayer> mlpFc1{}, mlpFc2{};
    uint32_t totalParams{0u};
};

// =====================================================================
//  64-byte aligned allocator (cache-line + AVX-512 / NEON friendly)
// =====================================================================
template <typename T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Align>; };

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}
    T* allocate(std::size_t n) {
        void* p = ::operator new(n * sizeof(T), std::align_val_t{Align});
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t{Align});
    }
    template <typename U>
    bool operator==(const AlignedAllocator<U, Align>&) const noexcept { return true; }
};
using AlignedVec = std::vector<float, AlignedAllocator<float, 64>>;

// =====================================================================
//  Model
// =====================================================================
struct Model {
    int vocabSize{0}, bosId{0};
    ModelOffsets off{};
    AlignedVec params{}, grads{}, m{}, v{};
    float beta1Pow{1.0F}, beta2Pow{1.0F};

    Expected<void> init(int vocabSizeIn, int bosIdIn) noexcept {
        vocabSize = vocabSizeIn; bosId = bosIdIn;
        uint32_t cursor{0u};
        auto mat = [](uint32_t r, uint32_t c) { return r * c; };
        off.wte = cursor;    cursor += mat(uint32_t(vocabSize), uint32_t(nEmbd));
        off.wpe = cursor;    cursor += mat(uint32_t(blockSize), uint32_t(nEmbd));
        off.lmHead = cursor; cursor += mat(uint32_t(vocabSize), uint32_t(nEmbd));
        for (int li = 0; li < nLayer; ++li) {
            auto L = toSize(li);
            off.attnWq[L] = cursor; cursor += mat(uint32_t(nEmbd), uint32_t(nEmbd));
            off.attnWk[L] = cursor; cursor += mat(uint32_t(nEmbd), uint32_t(nEmbd));
            off.attnWv[L] = cursor; cursor += mat(uint32_t(nEmbd), uint32_t(nEmbd));
            off.attnWo[L] = cursor; cursor += mat(uint32_t(nEmbd), uint32_t(nEmbd));
            off.mlpFc1[L] = cursor; cursor += mat(uint32_t(mlpHidden), uint32_t(nEmbd));
            off.mlpFc2[L] = cursor; cursor += mat(uint32_t(nEmbd), uint32_t(mlpHidden));
        }
        off.totalParams = cursor;
        size_t n = toSize(int(off.totalParams));
        params.assign(n, 0.0F); grads.assign(n, 0.0F);
        m.assign(n, 0.0F);      v.assign(n, 0.0F);
        beta1Pow = 1.0F; beta2Pow = 1.0F;
        return {};
    }

    void initParams(std::mt19937& rng) noexcept {
        std::normal_distribution<float> nd{0.0F, initStd};
        for (float& p : params) p = nd(rng);
    }
    void zeroGrads() noexcept { std::fill(grads.begin(), grads.end(), 0.0F); }

    void adamStep(int step, int numSteps) noexcept {
        const float lrT = learningRate * (1.0F - float(step) / float(numSteps));
        beta1Pow *= beta1; beta2Pow *= beta2;
        const float invBias1 = 1.0F / (1.0F - beta1Pow);
        const float invBias2 = 1.0F / (1.0F - beta2Pow);
        float* __restrict__ pp = params.data();
        float* __restrict__ gp = grads.data();
        float* __restrict__ mp = m.data();
        float* __restrict__ vp = v.data();
        const size_t n = params.size();
        for (size_t i = 0; i < n; ++i) {
            const float g = gp[i];
            mp[i] = beta1 * mp[i] + (1.0F - beta1) * g;
            vp[i] = beta2 * vp[i] + (1.0F - beta2) * (g * g);
            pp[i] -= lrT * (mp[i] * invBias1) / (std::sqrt(vp[i] * invBias2) + epsAdam);
        }
        zeroGrads();
    }
};

// =====================================================================
//  Templated math kernels
//
//  Compile-time dimensions let the compiler fully unroll inner loops
//  and emit tight SIMD.  __restrict__ removes aliasing barriers.
//
//  dot<N> uses 4 independent accumulators to break the FP dependency
//  chain — the CPU can retire 4 FMAs per cycle instead of stalling
//  on a single accumulator.  Requires N % 4 == 0 (static_assert above).
// =====================================================================

template <int N>
inline void setZero(float* __restrict__ a) noexcept {
    for (int i = 0; i < N; ++i) a[i] = 0.0F;
}

// 4-accumulator ILP dot product (from microgpt_turbo idea)
template <int N>
inline float dot(const float* __restrict__ a,
                 const float* __restrict__ b) noexcept {
    static_assert(N % 4 == 0, "dot<N> requires N divisible by 4");
    float acc0 = 0.0F, acc1 = 0.0F, acc2 = 0.0F, acc3 = 0.0F;
    for (int i = 0; i < N; i += 4) {
        acc0 += a[i + 0] * b[i + 0];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

// y = W x,  W:[NOut x NIn] row-major
template <int NOut, int NIn>
inline void linearForward(const float* __restrict__ w,
                          const float* __restrict__ x,
                          float* __restrict__ y) noexcept {
    for (int o = 0; o < NOut; ++o) {
        y[o] = dot<NIn>(w + o * NIn, x);
    }
}

// dW += dy outer x,  dx += W^T dy
template <int NOut, int NIn>
inline void linearBackwardAcc(const float* __restrict__ w,
                              const float* __restrict__ x,
                              const float* __restrict__ dy,
                              float* __restrict__ dW,
                              float* __restrict__ dx) noexcept {
    for (int o = 0; o < NOut; ++o) {
        const float dyo = dy[o];
        const float* __restrict__ row = w + o * NIn;
        float* __restrict__ dRow = dW + o * NIn;
        for (int i = 0; i < NIn; ++i) {
            dRow[i] += dyo * x[i];
            dx[i]   += row[i] * dyo;
        }
    }
}

template <int N>
inline float rmsnormForward(const float* __restrict__ x,
                            float* __restrict__ y) noexcept {
    float ms = dot<N>(x, x);  // reuse ILP dot for sum-of-squares
    ms = ms * (1.0F / float(N)) + rmsEps;
    const float inv = 1.0F / std::sqrt(ms);
    for (int i = 0; i < N; ++i) y[i] = x[i] * inv;
    return inv;
}

template <int N>
inline void rmsnormBackward(const float* __restrict__ dy,
                            const float* __restrict__ x,
                            float inv,
                            float* __restrict__ dx) noexcept {
    float dotDyX = dot<N>(dy, x);  // reuse ILP dot
    const float coeff = dotDyX * (inv * inv * inv) * (1.0F / float(N));
    for (int i = 0; i < N; ++i) dx[i] += dy[i] * inv - coeff * x[i];
}

// Fused LM-head forward (logits+softmax+CE) + backward
inline float lmHeadForwardBackward(const float* __restrict__ wLm,
                                   float* __restrict__ dWLm,
                                   const float* __restrict__ xOut,
                                   float* __restrict__ dXOut,
                                   float* __restrict__ logitsTmp,
                                   int vocabSize,
                                   int targetId,
                                   float scale_) noexcept {
    float maxVal = std::numeric_limits<float>::lowest();
    for (int o = 0; o < vocabSize; ++o) {
        float z = dot<nEmbd>(wLm + o * nEmbd, xOut);
        logitsTmp[toSize(o)] = z;
        maxVal = std::max(maxVal, z);
    }
    float sumExp = 0.0F;
    for (int o = 0; o < vocabSize; ++o) {
        float e = std::exp(logitsTmp[toSize(o)] - maxVal);
        logitsTmp[toSize(o)] = e;
        sumExp += e;
    }
    const float invSum = 1.0F / sumExp;
    const float pT = std::max(logitsTmp[toSize(targetId)] * invSum, 1e-12F);
    const float loss = -std::log(pT);
    setZero<nEmbd>(dXOut);
    for (int o = 0; o < vocabSize; ++o) {
        float dlog = logitsTmp[toSize(o)] * invSum;
        if (o == targetId) dlog -= 1.0F;
        dlog *= scale_;
        const float* __restrict__ row  = wLm  + o * nEmbd;
        float*       __restrict__ dRow = dWLm + o * nEmbd;
        for (int i = 0; i < nEmbd; ++i) {
            dRow[i]  += dlog * xOut[i];
            dXOut[i] += row[i] * dlog;
        }
    }
    return loss;
}

// =====================================================================
//  Training scratch - cache-friendly attention layout [t][h][s]
//
//  Original layout [t][s][h] had stride nHead on the inner s-loop.
//  New layout [t][h][s] gives stride 1, much better for vectorisation.
// =====================================================================
struct TrainScratch {
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> xEmbSum{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> x0{};
    alignas(64) std::array<float, blockSize> inv0{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> x1{};
    alignas(64) std::array<float, blockSize> inv1{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> q{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> k{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> v{};
    AlignedVec attnW{};   // [t][h][s] layout
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> attnConcat{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> attnProj{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> x2{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> x3{};
    alignas(64) std::array<float, blockSize> inv2{};
    alignas(64) std::array<std::array<float, mlpHidden>, blockSize> fc1Pre{};
    alignas(64) std::array<std::array<float, mlpHidden>, blockSize> fc1Relu{};
    alignas(64) std::array<std::array<uint8_t, mlpHidden>, blockSize> reluMask{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> fc2{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> xOut{};
    AlignedVec logitsTmp{};

    // Backward
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dXOut{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dX2{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dAttnProj{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dAttnConcat{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dX0{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dX1{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dQ{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dK{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dV{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dX3{};
    alignas(64) std::array<std::array<float, mlpHidden>, blockSize> dFc1Pre{};
    alignas(64) std::array<std::array<float, nEmbd>, blockSize> dFc2{};

    Expected<void> init(int vocabSize) noexcept {
        attnW.assign(toSize(blockSize * nHead * blockSize), 0.0F);
        logitsTmp.assign(toSize(vocabSize), 0.0F);
        return {};
    }
    // index = (t * nHead + h) * blockSize + s   => stride-1 on s
    float* awRow(int t, int h) noexcept {
        return &attnW[toSize((t * nHead + h) * blockSize)];
    }
    const float* awRow(int t, int h) const noexcept {
        return &attnW[toSize((t * nHead + h) * blockSize)];
    }
};

// =====================================================================
//  Attention forward / backward
// =====================================================================
inline void attentionForward(const TrainScratch& scIn,
                             TrainScratch& sc, int T) noexcept {
    for (int t = 0; t < T; ++t) {
        setZero<nEmbd>(sc.attnConcat[toSize(t)].data());
        for (int h = 0; h < nHead; ++h) {
            const int hs = h * headDim;
            float* __restrict__ aw = sc.awRow(t, h);
            float maxScore = std::numeric_limits<float>::lowest();
            for (int s = 0; s <= t; ++s) {
                float d = dot<headDim>(&scIn.q[toSize(t)][toSize(hs)],
                                       &scIn.k[toSize(s)][toSize(hs)]) * invSqrtHeadDim;
                aw[s] = d;
                maxScore = std::max(maxScore, d);
            }
            float sumExp = 0.0F;
            for (int s = 0; s <= t; ++s) {
                float e = std::exp(aw[s] - maxScore);
                aw[s] = e; sumExp += e;
            }
            const float invSum = 1.0F / sumExp;
            for (int s = 0; s <= t; ++s) aw[s] *= invSum;
            for (int j = 0; j < headDim; ++j) {
                float acc = 0.0F;
                for (int s = 0; s <= t; ++s)
                    acc += aw[s] * scIn.v[toSize(s)][toSize(hs + j)];
                sc.attnConcat[toSize(t)][toSize(hs + j)] = acc;
            }
        }
    }
}

inline void attentionBackward(const TrainScratch& sc,
                              TrainScratch& b, int T) noexcept {
    for (int t = 0; t < T; ++t) {
        setZero<nEmbd>(b.dQ[toSize(t)].data());
        setZero<nEmbd>(b.dK[toSize(t)].data());
        setZero<nEmbd>(b.dV[toSize(t)].data());
    }
    for (int h = 0; h < nHead; ++h) {
        const int hs = h * headDim;
        alignas(64) std::array<std::array<float, blockSize>, blockSize> dScore{};

        for (int t = 0; t < T; ++t) {
            const float* __restrict__ dHead = &b.dAttnConcat[toSize(t)][toSize(hs)];
            const float* __restrict__ aw = sc.awRow(t, h);
            for (int s = 0; s <= t; ++s) {
                const float w = aw[s];
                const float* __restrict__ vS = &sc.v[toSize(s)][toSize(hs)];
                float* __restrict__ dvS = &b.dV[toSize(s)][toSize(hs)];
                for (int j = 0; j < headDim; ++j) dvS[j] += w * dHead[j];
                dScore[toSize(t)][toSize(s)] = dot<headDim>(dHead, vS);
            }
        }
        for (int t = 0; t < T; ++t) {
            const float* __restrict__ aw = sc.awRow(t, h);
            float sumDwW = 0.0F;
            for (int s = 0; s <= t; ++s)
                sumDwW += dScore[toSize(t)][toSize(s)] * aw[s];
            for (int s = 0; s <= t; ++s)
                dScore[toSize(t)][toSize(s)] =
                    aw[s] * (dScore[toSize(t)][toSize(s)] - sumDwW);
        }
        for (int t = 0; t < T; ++t) {
            const float* __restrict__ qT  = &sc.q[toSize(t)][toSize(hs)];
            float*       __restrict__ dQT = &b.dQ[toSize(t)][toSize(hs)];
            for (int s = 0; s <= t; ++s) {
                const float ds = dScore[toSize(t)][toSize(s)] * invSqrtHeadDim;
                const float* __restrict__ kS  = &sc.k[toSize(s)][toSize(hs)];
                float*       __restrict__ dKS = &b.dK[toSize(s)][toSize(hs)];
                for (int j = 0; j < headDim; ++j) {
                    dQT[j] += ds * kS[j];
                    dKS[j] += ds * qT[j];
                }
            }
        }
    }
}

// =====================================================================
//  Training step
// =====================================================================
float trainStep(Model& model, TrainScratch& s, const Tokenizer& tok,
                const std::vector<std::string>& docs, int step,
                std::mt19937& /*rng*/, int& outTokenCount) noexcept {
    std::array<int, blockSize + 2> tokens{};
    tokens[0] = tok.bosId;
    const std::string& doc = docs[toSize(step % int(docs.size()))];
    int len = 1;
    for (char c : doc) {
        if (len >= blockSize + 1) break;
        auto idExp = tok.encodeChar(c);
        if (!idExp) return std::numeric_limits<float>::quiet_NaN();
        tokens[toSize(len)] = *idExp; ++len;
    }
    tokens[toSize(len)] = tok.bosId; ++len;
    const int T = std::min(blockSize, len - 1);
    outTokenCount = T;

    const float* __restrict__ pp = model.params.data();
    float*       __restrict__ gp = model.grads.data();

    // Forward: embedding + rmsnorm0
    for (int t = 0; t < T; ++t) {
        const int tid = tokens[toSize(t)];
        assign<nEmbd>(s.xEmbSum[toSize(t)].data(),
                      vec(pp + model.off.wte + tid * nEmbd) +
                          vec(pp + model.off.wpe + t * nEmbd));
        s.inv0[toSize(t)] = rmsnormForward<nEmbd>(
            s.xEmbSum[toSize(t)].data(), s.x0[toSize(t)].data());
    }

    // Forward: rmsnorm1 + QKV
    const float* __restrict__ wQ = pp + model.off.attnWq[0];
    const float* __restrict__ wK = pp + model.off.attnWk[0];
    const float* __restrict__ wV = pp + model.off.attnWv[0];
    for (int t = 0; t < T; ++t) {
        s.inv1[toSize(t)] = rmsnormForward<nEmbd>(
            s.x0[toSize(t)].data(), s.x1[toSize(t)].data());
        const float* __restrict__ x = s.x1[toSize(t)].data();
        linearForward<nEmbd, nEmbd>(wQ, x, s.q[toSize(t)].data());
        linearForward<nEmbd, nEmbd>(wK, x, s.k[toSize(t)].data());
        linearForward<nEmbd, nEmbd>(wV, x, s.v[toSize(t)].data());
    }

    attentionForward(s, s, T);

    // Forward: Wo + residual
    const float* __restrict__ wO = pp + model.off.attnWo[0];
    for (int t = 0; t < T; ++t) {
        linearForward<nEmbd, nEmbd>(wO, s.attnConcat[toSize(t)].data(),
                                    s.attnProj[toSize(t)].data());
        assign<nEmbd>(s.x2[toSize(t)].data(),
                      vec(s.x0[toSize(t)].data()) +
                          vec(s.attnProj[toSize(t)].data()));
    }

    // Forward: rmsnorm2 + MLP
    const float* __restrict__ wFc1 = pp + model.off.mlpFc1[0];
    const float* __restrict__ wFc2 = pp + model.off.mlpFc2[0];
    for (int t = 0; t < T; ++t) {
        s.inv2[toSize(t)] = rmsnormForward<nEmbd>(
            s.x2[toSize(t)].data(), s.x3[toSize(t)].data());
        linearForward<mlpHidden, nEmbd>(wFc1, s.x3[toSize(t)].data(),
                                        s.fc1Pre[toSize(t)].data());
        for (int i = 0; i < mlpHidden; ++i) {
            const float val = s.fc1Pre[toSize(t)][toSize(i)];
            const bool pos = val > 0.0F;
            s.fc1Relu[toSize(t)][toSize(i)] = pos ? val : 0.0F;
            s.reluMask[toSize(t)][toSize(i)] = uint8_t(pos);
        }
        linearForward<nEmbd, mlpHidden>(wFc2, s.fc1Relu[toSize(t)].data(),
                                        s.fc2[toSize(t)].data());
        assign<nEmbd>(s.xOut[toSize(t)].data(),
                      vec(s.x2[toSize(t)].data()) +
                          vec(s.fc2[toSize(t)].data()));
    }

    // ---- Backward ----
    for (int t = 0; t < T; ++t) {
        setZero<nEmbd>(s.dXOut[toSize(t)].data());
        setZero<nEmbd>(s.dX2[toSize(t)].data());
        setZero<nEmbd>(s.dAttnProj[toSize(t)].data());
        setZero<nEmbd>(s.dAttnConcat[toSize(t)].data());
        setZero<nEmbd>(s.dX0[toSize(t)].data());
        setZero<nEmbd>(s.dX1[toSize(t)].data());
        setZero<nEmbd>(s.dX3[toSize(t)].data());
        setZero<nEmbd>(s.dFc2[toSize(t)].data());
        setZero<mlpHidden>(s.dFc1Pre[toSize(t)].data());
    }

    // 1) Fused LM head
    float loss = 0.0F;
    const float* __restrict__ wLm  = pp + model.off.lmHead;
    float*       __restrict__ dWLm = gp + model.off.lmHead;
    const float invT = 1.0F / float(T);
    for (int t = 0; t < T; ++t)
        loss += lmHeadForwardBackward(wLm, dWLm, s.xOut[toSize(t)].data(),
                                      s.dXOut[toSize(t)].data(),
                                      s.logitsTmp.data(),
                                      model.vocabSize,
                                      tokens[toSize(t + 1)], invT);
    const float lossMean = loss * invT;

    // 2) dXOut -> dX2 + dFc2
    for (int t = 0; t < T; ++t) {
        accum<nEmbd>(s.dX2[toSize(t)].data(),
                     vec(s.dXOut[toSize(t)].data()));
        assign<nEmbd>(s.dFc2[toSize(t)].data(),
                      vec(s.dXOut[toSize(t)].data()));
    }

    // 3) fc2 backward
    {
        const float* __restrict__ w2  = pp + model.off.mlpFc2[0];
        float*       __restrict__ dW2 = gp + model.off.mlpFc2[0];
        for (int t = 0; t < T; ++t) {
            alignas(64) float dRelu[mlpHidden]{};
            for (int o = 0; o < nEmbd; ++o) {
                const float dyo = s.dFc2[toSize(t)][toSize(o)];
                const float* __restrict__ row  = w2  + o * mlpHidden;
                float*       __restrict__ dRow = dW2 + o * mlpHidden;
                for (int i = 0; i < mlpHidden; ++i) {
                    dRow[i]  += dyo * s.fc1Relu[toSize(t)][toSize(i)];
                    dRelu[i] += row[i] * dyo;
                }
            }
            for (int i = 0; i < mlpHidden; ++i)
                s.dFc1Pre[toSize(t)][toSize(i)] =
                    s.reluMask[toSize(t)][toSize(i)] ? dRelu[i] : 0.0F;
        }
    }

    // 4) fc1 backward
    {
        const float* __restrict__ w1  = pp + model.off.mlpFc1[0];
        float*       __restrict__ dW1 = gp + model.off.mlpFc1[0];
        for (int t = 0; t < T; ++t) {
            setZero<nEmbd>(s.dX3[toSize(t)].data());
            linearBackwardAcc<mlpHidden, nEmbd>(
                w1, s.x3[toSize(t)].data(), s.dFc1Pre[toSize(t)].data(),
                dW1, s.dX3[toSize(t)].data());
        }
    }

    // 5) rmsnorm2 backward
    for (int t = 0; t < T; ++t)
        rmsnormBackward<nEmbd>(s.dX3[toSize(t)].data(),
                               s.x2[toSize(t)].data(),
                               s.inv2[toSize(t)],
                               s.dX2[toSize(t)].data());

    // 6) residual split
    for (int t = 0; t < T; ++t) {
        accum<nEmbd>(s.dX0[toSize(t)].data(),
                     vec(s.dX2[toSize(t)].data()));
        accum<nEmbd>(s.dAttnProj[toSize(t)].data(),
                     vec(s.dX2[toSize(t)].data()));
    }

    // 7) Wo backward
    {
        float* __restrict__ dWO = gp + model.off.attnWo[0];
        for (int t = 0; t < T; ++t) {
            setZero<nEmbd>(s.dAttnConcat[toSize(t)].data());
            linearBackwardAcc<nEmbd, nEmbd>(
                wO, s.attnConcat[toSize(t)].data(),
                s.dAttnProj[toSize(t)].data(),
                dWO, s.dAttnConcat[toSize(t)].data());
        }
    }

    // 8) Attention backward
    attentionBackward(s, s, T);

    // 9) QKV backward
    for (int t = 0; t < T; ++t)
        setZero<nEmbd>(s.dX1[toSize(t)].data());
    {
        float* __restrict__ dWQ = gp + model.off.attnWq[0];
        float* __restrict__ dWK = gp + model.off.attnWk[0];
        float* __restrict__ dWV = gp + model.off.attnWv[0];
        for (int t = 0; t < T; ++t) {
            float* __restrict__ dx1 = s.dX1[toSize(t)].data();
            const float* __restrict__ x1t = s.x1[toSize(t)].data();
            linearBackwardAcc<nEmbd, nEmbd>(wQ, x1t,
                                            s.dQ[toSize(t)].data(), dWQ, dx1);
            linearBackwardAcc<nEmbd, nEmbd>(wK, x1t,
                                            s.dK[toSize(t)].data(), dWK, dx1);
            linearBackwardAcc<nEmbd, nEmbd>(wV, x1t,
                                            s.dV[toSize(t)].data(), dWV, dx1);
        }
    }

    // 10) rmsnorm1 backward
    for (int t = 0; t < T; ++t)
        rmsnormBackward<nEmbd>(s.dX1[toSize(t)].data(),
                               s.x0[toSize(t)].data(),
                               s.inv1[toSize(t)],
                               s.dX0[toSize(t)].data());

    // 11) rmsnorm0 backward -> embedding grads
    for (int t = 0; t < T; ++t) {
        alignas(64) float dXEmbSum[nEmbd]{};
        rmsnormBackward<nEmbd>(s.dX0[toSize(t)].data(),
                               s.xEmbSum[toSize(t)].data(),
                               s.inv0[toSize(t)], dXEmbSum);
        const int tid = tokens[toSize(t)];
        float* __restrict__ gWte = gp + model.off.wte + tid * nEmbd;
        float* __restrict__ gWpe = gp + model.off.wpe + t * nEmbd;
        for (int i = 0; i < nEmbd; ++i) {
            gWte[i] += dXEmbSum[i];
            gWpe[i] += dXEmbSum[i];
        }
    }
    return lossMean;
}

// =====================================================================
//  Inference
// =====================================================================
int sampleCategorical(const std::vector<float>& probs,
                      std::mt19937& rng) noexcept {
    float sum = 0.0F;
    for (float p : probs) sum += p;
    if (!(sum > 0.0F)) {
        std::uniform_int_distribution<int> uid{0, int(probs.size()) - 1};
        return uid(rng);
    }
    std::uniform_real_distribution<float> urd{0.0F, sum};
    float r = urd(rng), cdf = 0.0F;
    for (int i = 0; i < int(probs.size()); ++i) {
        cdf += probs[toSize(i)];
        if (r <= cdf) return i;
    }
    return int(probs.size()) - 1;
}

struct InferScratch {
    alignas(64) std::array<float, nEmbd> xEmbSum{};
    alignas(64) std::array<float, nEmbd> x0{};
    alignas(64) std::array<float, nEmbd> x1{};
    alignas(64) std::array<float, nEmbd> q{};
    alignas(64) std::array<float, nEmbd> k{};
    alignas(64) std::array<float, nEmbd> v{};
    alignas(64) std::array<float, nEmbd> attnConcat{};
    alignas(64) std::array<float, nEmbd> attnProj{};
    alignas(64) std::array<float, nEmbd> x2{};
    alignas(64) std::array<float, nEmbd> x3{};
    alignas(64) std::array<float, mlpHidden> fc1{};
    alignas(64) std::array<float, nEmbd> fc2{};
    alignas(64) std::array<float, nEmbd> xOut{};
    std::vector<float> logits{};
    std::vector<float> probs{};
    Expected<void> init(int vocabSize) noexcept {
        logits.assign(toSize(vocabSize), 0.0F);
        probs.assign(toSize(vocabSize), 0.0F);
        return {};
    }
};

void runInference(const Model& model, const Tokenizer& tok,
                  int numSamples, float temperature,
                  std::mt19937& rng) {
    std::cout << "\n--- inference (new, hallucinated names) ---\n";
    InferScratch s{};
    (void)s.init(model.vocabSize);

    alignas(64) std::array<std::array<std::array<float, nEmbd>,
                                      blockSize>, nLayer> kCache{};
    alignas(64) std::array<std::array<std::array<float, nEmbd>,
                                      blockSize>, nLayer> vCache{};
    std::array<int, nLayer> cacheLen{};
    const float* __restrict__ pp = model.params.data();
    uint64_t totalGeneratedChars = 0u;

    for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx) {
        for (int li = 0; li < nLayer; ++li) cacheLen[toSize(li)] = 0;
        int tokenId = tok.bosId;
        std::array<char, blockSize + 1> out{};
        int outLen = 0;

        for (int posId = 0; posId < blockSize; ++posId) {
            assign<nEmbd>(s.xEmbSum.data(),
                          vec(pp + model.off.wte + tokenId * nEmbd) +
                              vec(pp + model.off.wpe + posId * nEmbd));
            (void)rmsnormForward<nEmbd>(s.xEmbSum.data(), s.x0.data());
            (void)rmsnormForward<nEmbd>(s.x0.data(), s.x1.data());

            linearForward<nEmbd, nEmbd>(pp + model.off.attnWq[0],
                                        s.x1.data(), s.q.data());
            linearForward<nEmbd, nEmbd>(pp + model.off.attnWk[0],
                                        s.x1.data(), s.k.data());
            linearForward<nEmbd, nEmbd>(pp + model.off.attnWv[0],
                                        s.x1.data(), s.v.data());

            const int t = cacheLen[0];
            for (int i = 0; i < nEmbd; ++i) {
                kCache[0][toSize(t)][toSize(i)] = s.k[toSize(i)];
                vCache[0][toSize(t)][toSize(i)] = s.v[toSize(i)];
            }
            cacheLen[0] = t + 1;

            setZero<nEmbd>(s.attnConcat.data());
            for (int h = 0; h < nHead; ++h) {
                const int hs = h * headDim;
                float maxScore = std::numeric_limits<float>::lowest();
                alignas(64) std::array<float, blockSize> score;
                for (int ss = 0; ss < cacheLen[0]; ++ss) {
                    float d = dot<headDim>(
                                  &s.q[toSize(hs)],
                                  &kCache[0][toSize(ss)][toSize(hs)]) * invSqrtHeadDim;
                    score[toSize(ss)] = d;
                    maxScore = std::max(maxScore, d);
                }
                alignas(64) std::array<float, blockSize> w;
                float sumExp = 0.0F;
                for (int ss = 0; ss < cacheLen[0]; ++ss) {
                    float e = std::exp(score[toSize(ss)] - maxScore);
                    w[toSize(ss)] = e; sumExp += e;
                }
                const float invSum = 1.0F / sumExp;
                for (int ss = 0; ss < cacheLen[0]; ++ss)
                    w[toSize(ss)] *= invSum;
                for (int j = 0; j < headDim; ++j) {
                    float acc = 0.0F;
                    for (int ss = 0; ss < cacheLen[0]; ++ss)
                        acc += w[toSize(ss)] *
                               vCache[0][toSize(ss)][toSize(hs + j)];
                    s.attnConcat[toSize(hs + j)] = acc;
                }
            }

            linearForward<nEmbd, nEmbd>(pp + model.off.attnWo[0],
                                        s.attnConcat.data(),
                                        s.attnProj.data());
            assign<nEmbd>(s.x2.data(),
                          vec(s.x0.data()) + vec(s.attnProj.data()));
            (void)rmsnormForward<nEmbd>(s.x2.data(), s.x3.data());

            linearForward<mlpHidden, nEmbd>(pp + model.off.mlpFc1[0],
                                            s.x3.data(), s.fc1.data());
            for (int i = 0; i < mlpHidden; ++i)
                s.fc1[toSize(i)] =
                    (s.fc1[toSize(i)] > 0.0F) ? s.fc1[toSize(i)] : 0.0F;
            linearForward<nEmbd, mlpHidden>(pp + model.off.mlpFc2[0],
                                            s.fc1.data(), s.fc2.data());
            assign<nEmbd>(s.xOut.data(),
                          vec(s.x2.data()) + vec(s.fc2.data()));

            const float* __restrict__ wLm = pp + model.off.lmHead;
            for (int o = 0; o < model.vocabSize; ++o)
                s.logits[toSize(o)] =
                    dot<nEmbd>(wLm + o * nEmbd, s.xOut.data());

            float maxVal = std::numeric_limits<float>::lowest();
            for (int i = 0; i < model.vocabSize; ++i)
                maxVal = std::max(maxVal,
                                  s.logits[toSize(i)] / temperature);
            float sumExp = 0.0F;
            for (int i = 0; i < model.vocabSize; ++i) {
                float e = std::exp(
                    s.logits[toSize(i)] / temperature - maxVal);
                s.probs[toSize(i)] = e; sumExp += e;
            }
            const float invSum = 1.0F / sumExp;
            for (int i = 0; i < model.vocabSize; ++i)
                s.probs[toSize(i)] *= invSum;

            if (posId == 0) {
                s.probs[toSize(tok.bosId)] = 0.0F;
                float reSum = 0.0F;
                for (int i = 0; i < model.vocabSize; ++i)
                    reSum += s.probs[toSize(i)];
                if (reSum > 0.0F) {
                    const float reInv = 1.0F / reSum;
                    for (int i = 0; i < model.vocabSize; ++i)
                        s.probs[toSize(i)] *= reInv;
                }
            }
            tokenId = sampleCategorical(s.probs, rng);
            if (tokenId == tok.bosId) break;
            out[toSize(outLen)] = tok.decodeId(tokenId); ++outLen;
        }
        out[toSize(outLen)] = '\0';
        totalGeneratedChars += uint64_t(outLen);
        std::cout << "sample " << std::setw(2) << (sampleIdx + 1)
                  << ": " << out.data() << "\n";
    }
    std::cout << "\n";
    std::cout << std::left << std::setw(34) << "generated chars (total)"
              << std::right << std::setw(12) << totalGeneratedChars << "\n";
}

// =====================================================================
//  Args + main
// =====================================================================
struct Args {
    std::string_view datasetPath{"input.txt"};
    int steps{defaultSteps};
    int samples{defaultSamples};
    int logEvery{defaultLogEvery};
    float temperature{defaultTemperature};
};

Args parseArgs(int argc, char** argv) noexcept {
    Args a{};
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i]
                                   ? std::string_view{argv[i]} : std::string_view{};
        if (arg == "--steps" && (i + 1) < argc) {
            a.steps = std::max(1,
                               parseInt(std::string_view{argv[i + 1]}, a.steps));
            i += 1;
        } else if (arg == "--samples" && (i + 1) < argc) {
            a.samples = std::max(1,
                                 parseInt(std::string_view{argv[i + 1]}, a.samples));
            i += 1;
        } else if (arg == "--log-every" && (i + 1) < argc) {
            a.logEvery = std::max(1,
                                  parseInt(std::string_view{argv[i + 1]}, a.logEvery));
            i += 1;
        } else if (arg == "--temperature" && (i + 1) < argc) {
            a.temperature = std::max(1e-6F,
                                     parseFloat(std::string_view{argv[i + 1]}, a.temperature));
            i += 1;
        } else if (!arg.empty() && arg[0] != '-') {
            a.datasetPath = arg;
        }
    }
    return a;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        const auto tProgramStart = Clock::now();
        const Args args = parseArgs(argc, argv);
        std::mt19937 rng{42u};
        const auto tInitStart = Clock::now();

        auto docsExp = readDocs(args.datasetPath);
        if (!docsExp) {
            std::cerr << "error: " << docsExp.error().message
                      << " (" << args.datasetPath << ")\n";
            return 1;
        }
        auto docs = std::move(*docsExp);
        std::shuffle(docs.begin(), docs.end(), rng);
        std::cout << "num docs: " << docs.size() << "\n";

        auto tokExp = Tokenizer::build(docs);
        if (!tokExp) {
            std::cerr << "error: tokenizer build failed\n";
            return 1;
        }
        const Tokenizer tok = *tokExp;
        std::cout << "vocab size: " << tok.vocabSize << "\n";

        Model model{};
        if (!model.init(tok.vocabSize, tok.bosId)) {
            std::cerr << "error: model init failed\n";
            return 1;
        }
        model.initParams(rng);
        std::cout << "num params: " << model.params.size() << "\n";

        TrainScratch scratch{};
        if (!scratch.init(model.vocabSize)) {
            std::cerr << "error: scratch init failed\n";
            return 1;
        }

        const auto tInitEnd = Clock::now();
        const auto tTrainStart = Clock::now();
        float lossFirst = 0.0F, lossLast = 0.0F;
        float lossMin = std::numeric_limits<float>::max();
        float lossSum = 0.0F;
        int lastTokenCount = 0;

        for (int step = 0; step < args.steps; ++step) {
            float loss = trainStep(model, scratch, tok, docs,
                                   step, rng, lastTokenCount);
            model.adamStep(step, args.steps);
            if (step == 0) lossFirst = loss;
            lossLast = loss;
            lossMin = std::min(lossMin, loss);
            lossSum += loss;
            const bool shouldLog =
                ((step + 1) % args.logEvery == 0) ||
                (step + 1) == 1 || (step + 1) == args.steps;
            if (shouldLog)
                std::cout << "step " << std::setw(4) << (step + 1)
                          << " / " << std::setw(4) << args.steps
                          << " | loss " << std::fixed
                          << std::setprecision(4) << loss << "\n";
        }

        const auto tTrainEnd = Clock::now();
        const auto tInferStart = Clock::now();
        runInference(model, tok, args.samples,
                     args.temperature, rng);
        const auto tInferEnd = Clock::now();
        const auto tProgramEnd = Clock::now();

        const auto initDur  = tInitEnd - tInitStart;
        const auto trainDur = tTrainEnd - tTrainStart;
        const auto inferDur = tInferEnd - tInferStart;
        const auto totalDur = tProgramEnd - tProgramStart;
        const float lossAvg = lossSum / float(args.steps);

        const auto printKey = [](std::string_view key) {
            std::cout << std::left << std::setw(34) << key;
        };
        const auto printMs = [&](std::string_view key,
                                 Clock::duration d) {
            printKey(key);
            std::cout << std::right << std::setw(12) << std::fixed
                      << std::setprecision(2)
                      << toMilliseconds(d) << " ms\n";
        };
        const auto printUs = [&](std::string_view key, double us) {
            printKey(key);
            std::cout << std::right << std::setw(12) << std::fixed
                      << std::setprecision(2) << us << " us\n";
        };

        std::cout << "\n--- run statistics ---\n";
        printKey("mode");
        std::cout << "turbo-kernels\n";
        printKey("dataset path");
        std::cout << args.datasetPath << "\n";
        printKey("steps");
        std::cout << args.steps << "\n";
        printKey("tokens per step (last)");
        std::cout << lastTokenCount << "\n";
        printMs("init (includes allocations)", initDur);
        printMs("training", trainDur);
        printMs("inference", inferDur);
        printMs("total (init+train+infer)", totalDur);
        printMs("train+infer (no init)", trainDur + inferDur);
        printUs("training per step",
                toMicroseconds(trainDur) / double(args.steps));
        printUs("inference per sample",
                toMicroseconds(inferDur) / double(args.samples));
        printKey("loss");
        std::cout << std::fixed << std::setprecision(4)
                  << "first=" << lossFirst
                  << " last=" << lossLast
                  << " min=" << lossMin
                  << " avg=" << lossAvg << "\n";
        return 0;
    } catch (const std::bad_alloc&) {
        std::cerr << "fatal: out of memory during initialization\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "fatal init exception: " << e.what() << "\n";
        return 1;
    }
}
