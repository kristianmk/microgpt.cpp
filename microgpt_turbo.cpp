// Written by K. M. Knausgård 2026-02-14, based on https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
//
// C++23, std-only. Max-performance CPU kernels (no autograd graph).
//
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

// ----------------------------- model constants -----------------------------
constexpr int nEmbd{16};
constexpr int nHead{4};
constexpr int nLayer{1};
constexpr int blockSize{16};

static_assert(nEmbd > 0);
static_assert(nHead > 0);
static_assert(nLayer > 0);
static_assert(blockSize > 0);
static_assert(nEmbd % nHead == 0);

constexpr int headDim{nEmbd / nHead};

constexpr float initStd{0.08F};
constexpr float rmsEps{1e-5F};

constexpr float learningRate{0.01F};
constexpr float beta1{0.85F};
constexpr float beta2{0.99F};
constexpr float epsAdam{1e-8F};

constexpr int defaultSteps{1000};
constexpr int defaultSamples{20};
constexpr int defaultLogEvery{25};
constexpr float defaultTemperature{0.5F};

// ----------------------------- errors -----------------------------
enum class ErrorCode : uint8_t {
    fileOpenFailed,
    fileReadFailed,
    emptyDataset,
    invalidChar
};

struct Error {
    ErrorCode code{};
    std::string_view message{};
};

template <class T>
using Expected = std::expected<T, Error>;

// ----------------------------- utilities -----------------------------
inline size_t toSize(int v) noexcept {
    return static_cast<size_t>(v);
}

inline uint32_t toU32(int v) noexcept {
    return static_cast<uint32_t>(v);
}

using Clock = std::chrono::steady_clock;

inline double toMilliseconds(Clock::duration d) noexcept {
    return std::chrono::duration<double, std::milli>{d}.count();
}

inline double toMicroseconds(Clock::duration d) noexcept {
    return std::chrono::duration<double, std::micro>{d}.count();
}

inline int parseInt(std::string_view s, int fallback) noexcept {
    int v{fallback};
    const auto* b{s.data()};
    const auto* e{s.data() + s.size()};
    const auto res{std::from_chars(b, e, v)};
    if (res.ec != std::errc{} || res.ptr != e) {
        return fallback;
    }
    return v;
}

// Simple, predictable float parser: sign? digits? ('.' digits?) . No exponent.
// Good enough for temperature.
inline float parseFloat(std::string_view s, float fallback) noexcept {
    if (s.empty()) {
        return fallback;
    }

    int sign{1};
    size_t i{0};

    if (s[0] == '-') {
        sign = -1;
        i = 1;
    } else if (s[0] == '+') {
        i = 1;
    }

    float intPart{0.0F};
    bool any{false};

    for (; i < s.size(); ++i) {
        const char c{s[i]};
        if (c >= '0' && c <= '9') {
            any = true;
            intPart = intPart * 10.0F + float(c - '0');
        } else {
            break;
        }
    }

    float fracPart{0.0F};
    float fracScale{1.0F};

    if (i < s.size() && s[i] == '.') {
        i += 1;
        for (; i < s.size(); ++i) {
            const char c{s[i]};
            if (c >= '0' && c <= '9') {
                any = true;
                fracScale *= 10.0F;
                fracPart = fracPart * 10.0F + float(c - '0');
            } else {
                return fallback;
            }
        }
    } else {
        if (i != s.size()) {
            return fallback;
        }
    }

    if (!any) {
        return fallback;
    }

    const float v{(intPart + (fracPart / fracScale)) * float(sign)};
    return v;
}

// ----------------------------- dataset/tokenizer -----------------------------
Expected<std::vector<std::string>> readDocs(std::string_view path) noexcept {
    std::ifstream in{std::string{path}, std::ios::binary};
    if (!in.is_open()) {
        return std::unexpected(Error{ErrorCode::fileOpenFailed, "failed to open dataset file"});
    }

    std::vector<std::string> docs{};
    std::string line{};

    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty()) {
            docs.push_back(line);
        }
    }

    if (!in.good() && !in.eof()) {
        return std::unexpected(Error{ErrorCode::fileReadFailed, "failed while reading dataset file"});
    }
    if (docs.empty()) {
        return std::unexpected(Error{ErrorCode::emptyDataset, "dataset is empty"});
    }

    return docs;
}

struct Tokenizer {
    std::string uchars{};
    std::array<int, 256> charToId{};
    int bosId{-1};
    int vocabSize{0};

    static Expected<Tokenizer> build(const std::vector<std::string>& docs) noexcept {
        Tokenizer t{};
        t.charToId.fill(-1);

        std::array<bool, 256> seen{};
        seen.fill(false);

        for (const auto& d : docs) {
            for (unsigned char c : d) {
                seen[c] = true;
            }
        }

        t.uchars.clear();
        t.uchars.reserve(256);

        for (int c{0}; c < 256; ++c) {
            if (seen[toSize(c)]) {
                t.uchars.push_back(static_cast<char>(c));
            }
        }

        std::sort(t.uchars.begin(), t.uchars.end());

        for (int i{0}; i < static_cast<int>(t.uchars.size()); ++i) {
            t.charToId[static_cast<unsigned char>(t.uchars[toSize(i)])] = i;
        }

        t.bosId = static_cast<int>(t.uchars.size());
        t.vocabSize = t.bosId + 1;
        return t;
    }

    Expected<int> encodeChar(char c) const noexcept {
        const int id{charToId[static_cast<unsigned char>(c)]};
        if (id < 0) {
            return std::unexpected(Error{ErrorCode::invalidChar, "encountered char not in vocab"});
        }
        return id;
    }

    char decodeId(int id) const noexcept {
        return uchars[toSize(id)];
    }
};

// ----------------------------- model params layout -----------------------------
struct ModelOffsets {
    uint32_t wte{0u};
    uint32_t wpe{0u};
    uint32_t lmHead{0u};

    std::array<uint32_t, nLayer> attnWq{};
    std::array<uint32_t, nLayer> attnWk{};
    std::array<uint32_t, nLayer> attnWv{};
    std::array<uint32_t, nLayer> attnWo{};

    std::array<uint32_t, nLayer> mlpFc1{};
    std::array<uint32_t, nLayer> mlpFc2{};

    uint32_t totalParams{0u};
};

struct Model {
    int vocabSize{0};
    int bosId{0};
    ModelOffsets off{};

    std::vector<float> params{};
    std::vector<float> grads{};
    std::vector<float> m{};
    std::vector<float> v{};

    Expected<void> init(int vocabSizeIn, int bosIdIn) noexcept {
        vocabSize = vocabSizeIn;
        bosId = bosIdIn;

        uint32_t cursor{0u};
        const auto mat = [](uint32_t nOut, uint32_t nIn) noexcept -> uint32_t {
            return nOut * nIn;
        };

        off.wte = cursor;
        cursor += mat(static_cast<uint32_t>(vocabSize), static_cast<uint32_t>(nEmbd));

        off.wpe = cursor;
        cursor += mat(static_cast<uint32_t>(blockSize), static_cast<uint32_t>(nEmbd));

        off.lmHead = cursor;
        cursor += mat(static_cast<uint32_t>(vocabSize), static_cast<uint32_t>(nEmbd));

        for (int li{0}; li < nLayer; ++li) {
            off.attnWq[toSize(li)] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.attnWk[toSize(li)] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.attnWv[toSize(li)] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.attnWo[toSize(li)] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.mlpFc1[toSize(li)] = cursor;
            cursor += mat(static_cast<uint32_t>(4 * nEmbd), static_cast<uint32_t>(nEmbd));

            off.mlpFc2[toSize(li)] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(4 * nEmbd));
        }

        off.totalParams = cursor;

        params.assign(toSize(static_cast<int>(off.totalParams)), 0.0F);
        grads.assign(toSize(static_cast<int>(off.totalParams)), 0.0F);
        m.assign(toSize(static_cast<int>(off.totalParams)), 0.0F);
        v.assign(toSize(static_cast<int>(off.totalParams)), 0.0F);

        return {};
    }

    void initParams(std::mt19937& rng) noexcept {
        std::normal_distribution<float> nd{0.0F, initStd};
        for (float& p : params) {
            p = nd(rng);
        }
    }

    void zeroGrads() noexcept {
        std::fill(grads.begin(), grads.end(), 0.0F);
    }

    void adamStep(int step, int numSteps) noexcept {
        const float lrT{learningRate * (1.0F - float(step) / float(numSteps))};

        const float b1t{std::pow(beta1, float(step + 1))};
        const float b2t{std::pow(beta2, float(step + 1))};

        for (size_t i{0}; i < params.size(); ++i) {
            const float g{grads[i]};

            m[i] = beta1 * m[i] + (1.0F - beta1) * g;
            v[i] = beta2 * v[i] + (1.0F - beta2) * (g * g);

            const float mHat{m[i] / (1.0F - b1t)};
            const float vHat{v[i] / (1.0F - b2t)};

            params[i] -= lrT * mHat / (std::sqrt(vHat) + epsAdam);
        }

        zeroGrads();
    }
};

// ----------------------------- math kernels -----------------------------
inline void addVec(const float* a, const float* b, float* out, int n) noexcept {
    for (int i{0}; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

inline void addInPlace(float* a, const float* b, int n) noexcept {
    for (int i{0}; i < n; ++i) {
        a[i] += b[i];
    }
}

inline void setZero(float* a, int n) noexcept {
    for (int i{0}; i < n; ++i) {
        a[i] = 0.0F;
    }
}

inline float dot(const float* a, const float* b, int n) noexcept {
    float acc{0.0F};
    for (int i{0}; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

// y = W x, W: [nOut][nIn] row-major
inline void linearForward(const float* w, const float* x, float* y, int nOut, int nIn) noexcept {
    for (int o{0}; o < nOut; ++o) {
        const float* row{w + o * nIn};
        y[o] = dot(row, x, nIn);
    }
}

// Backward for y = W x
// dW += dy ⊗ x
// dx += W^T dy
inline void linearBackwardAcc(const float* w,
                              const float* x,
                              const float* dy,
                              float* dW,
                              float* dx,
                              int nOut,
                              int nIn) noexcept {
    for (int o{0}; o < nOut; ++o) {
        const float dyo{dy[o]};
        const float* row{w + o * nIn};
        float* dRow{dW + o * nIn};

        for (int i{0}; i < nIn; ++i) {
            dRow[i] += dyo * x[i];
            dx[i] += row[i] * dyo;
        }
    }
}

// RMSNorm: y = x * inv, inv = 1/sqrt(mean(x^2) + eps)
inline float rmsnormForward(const float* x, float* y, int n) noexcept {
    float ms{0.0F};
    for (int i{0}; i < n; ++i) {
        ms += x[i] * x[i];
    }
    ms *= 1.0F / float(n);
    ms += rmsEps;

    const float inv{1.0F / std::sqrt(ms)};
    for (int i{0}; i < n; ++i) {
        y[i] = x[i] * inv;
    }
    return inv;
}

// Backward RMSNorm.
// Given dy, x, inv => dx.
// dx_i = dy_i*inv - (dot(dy, x) * (inv^3 / n)) * x_i
inline void rmsnormBackward(const float* dy, const float* x, float inv, float* dx, int n) noexcept {
    float dotDyX{0.0F};
    for (int i{0}; i < n; ++i) {
        dotDyX += dy[i] * x[i];
    }

    const float inv3{inv * inv * inv};
    const float coeff{dotDyX * (inv3 / float(n))};

    for (int i{0}; i < n; ++i) {
        dx[i] += dy[i] * inv - coeff * x[i];
    }
}

// Stable softmax into probs (len = L). Returns -log(prob[target]) for CE.
inline float softmaxCrossEntropyForward(const float* logits, float* probs, int L, int target) noexcept {
    float maxVal{-std::numeric_limits<float>::infinity()};
    for (int i{0}; i < L; ++i) {
        maxVal = std::max(maxVal, logits[i]);
    }

    float sumExp{0.0F};
    for (int i{0}; i < L; ++i) {
        const float e{std::exp(logits[i] - maxVal)};
        probs[i] = e;
        sumExp += e;
    }

    const float invSum{1.0F / sumExp};
    for (int i{0}; i < L; ++i) {
        probs[i] *= invSum;
    }

    const float p{std::max(probs[target], 1e-12F)};
    return -std::log(p);
}

// ----------------------------- training scratch -----------------------------
struct TrainScratch {
    // Forward activations (for T <= blockSize)
    std::array<std::array<float, nEmbd>, blockSize> xEmbSum{};
    std::array<std::array<float, nEmbd>, blockSize> x0{};
    std::array<float, blockSize> inv0{};

    std::array<std::array<float, nEmbd>, blockSize> x1{};
    std::array<float, blockSize> inv1{};

    std::array<std::array<float, nEmbd>, blockSize> q{};
    std::array<std::array<float, nEmbd>, blockSize> k{};
    std::array<std::array<float, nEmbd>, blockSize> v{};

    // Attention weights: [t][s][h] stored in flat buffer size blockSize*blockSize*nHead
    std::vector<float> attnW{};

    std::array<std::array<float, nEmbd>, blockSize> attnConcat{};
    std::array<std::array<float, nEmbd>, blockSize> attnProj{};
    std::array<std::array<float, nEmbd>, blockSize> x2{};

    std::array<std::array<float, nEmbd>, blockSize> x3{};
    std::array<float, blockSize> inv2{};

    std::array<std::array<float, 4 * nEmbd>, blockSize> fc1Pre{};
    std::array<std::array<float, 4 * nEmbd>, blockSize> fc1Relu{};
    std::array<std::array<uint8_t, 4 * nEmbd>, blockSize> reluMask{};

    std::array<std::array<float, nEmbd>, blockSize> fc2{};
    std::array<std::array<float, nEmbd>, blockSize> xOut{};

    // probs per position: [T][vocab]
    std::vector<float> probs{};

    // Temporary buffers (reused)
    std::vector<float> logitsTmp{};

    // Backward buffers
    std::array<std::array<float, nEmbd>, blockSize> dXOut{};
    std::array<std::array<float, nEmbd>, blockSize> dX2{};
    std::array<std::array<float, nEmbd>, blockSize> dAttnProj{};
    std::array<std::array<float, nEmbd>, blockSize> dAttnConcat{};
    std::array<std::array<float, nEmbd>, blockSize> dX0{};
    std::array<std::array<float, nEmbd>, blockSize> dX1{};
    std::array<std::array<float, nEmbd>, blockSize> dQ{};
    std::array<std::array<float, nEmbd>, blockSize> dK{};
    std::array<std::array<float, nEmbd>, blockSize> dV{};

    std::array<std::array<float, nEmbd>, blockSize> dX3{};
    std::array<std::array<float, 4 * nEmbd>, blockSize> dFc1Pre{};
    std::array<std::array<float, 4 * nEmbd>, blockSize> dFc1Relu{};
    std::array<std::array<float, nEmbd>, blockSize> dFc2{};

    Expected<void> init(int vocabSize) noexcept {
        attnW.assign(toSize(blockSize * blockSize * nHead), 0.0F);
        probs.assign(toSize(blockSize * vocabSize), 0.0F);
        logitsTmp.assign(toSize(vocabSize), 0.0F);
        return {};
    }

    inline float& attnWeight(int t, int s, int h) noexcept {
        return attnW[toSize(((t * blockSize + s) * nHead) + h)];
    }

    inline const float& attnWeight(int t, int s, int h) const noexcept {
        return attnW[toSize(((t * blockSize + s) * nHead) + h)];
    }

    inline float* probsRow(int t, int vocabSize) noexcept {
        return probs.data() + toSize(t * vocabSize);
    }

    inline const float* probsRow(int t, int vocabSize) const noexcept {
        return probs.data() + toSize(t * vocabSize);
    }
};

// ----------------------------- attention forward/backward -----------------------------
inline void attentionForward(const TrainScratch& scIn,
                             TrainScratch& sc,
                             int T) noexcept {
    // Uses q,k,v from sc.q/sc.k/sc.v, writes attnConcat.
    const float invSqrtHd{1.0F / std::sqrt(float(headDim))};

    for (int t{0}; t < T; ++t) {
        for (int i{0}; i < nEmbd; ++i) {
            sc.attnConcat[toSize(t)][toSize(i)] = 0.0F;
        }

        for (int h{0}; h < nHead; ++h) {
            const int hs{h * headDim};

            // scores s=0..t
            float maxScore{-std::numeric_limits<float>::infinity()};
            std::array<float, blockSize> score{};
            for (int s{0}; s <= t; ++s) {
                const float d{dot(&scIn.q[toSize(t)][toSize(hs)], &scIn.k[toSize(s)][toSize(hs)], headDim) * invSqrtHd};
                score[toSize(s)] = d;
                maxScore = std::max(maxScore, d);
            }

            float sumExp{0.0F};
            for (int s{0}; s <= t; ++s) {
                const float e{std::exp(score[toSize(s)] - maxScore)};
                sc.attnWeight(t, s, h) = e;
                sumExp += e;
            }

            const float invSum{1.0F / sumExp};
            for (int s{0}; s <= t; ++s) {
                sc.attnWeight(t, s, h) *= invSum;
            }

            // head output
            for (int j{0}; j < headDim; ++j) {
                float acc{0.0F};
                for (int s{0}; s <= t; ++s) {
                    acc += sc.attnWeight(t, s, h) * scIn.v[toSize(s)][toSize(hs + j)];
                }
                sc.attnConcat[toSize(t)][toSize(hs + j)] = acc;
            }
        }
    }
}

inline void attentionBackward(const TrainScratch& sc,
                              TrainScratch& b,
                              int T) noexcept {
    const float invSqrtHd{1.0F / std::sqrt(float(headDim))};

    for (int t{0}; t < T; ++t) {
        for (int i{0}; i < nEmbd; ++i) {
            b.dQ[toSize(t)][toSize(i)] = 0.0F;
            b.dK[toSize(t)][toSize(i)] = 0.0F;
            b.dV[toSize(t)][toSize(i)] = 0.0F;
        }
    }

    // Per head
    for (int h{0}; h < nHead; ++h) {
        const int hs{h * headDim};

        std::array<std::array<float, blockSize>, blockSize> dScore{};
        for (int t{0}; t < T; ++t) {
            for (int s{0}; s < T; ++s) {
                dScore[toSize(t)][toSize(s)] = 0.0F;
            }
        }

        // 1) dV and dW from headOut = sum_s w_ts * v_s
        for (int t{0}; t < T; ++t) {
            const float* dHead{&b.dAttnConcat[toSize(t)][toSize(hs)]};

            for (int s{0}; s <= t; ++s) {
                const float w{sc.attnWeight(t, s, h)};
                const float* vS{&sc.v[toSize(s)][toSize(hs)]};

                for (int j{0}; j < headDim; ++j) {
                    b.dV[toSize(s)][toSize(hs + j)] += w * dHead[j];
                }

                const float dW{dot(dHead, vS, headDim)};
                dScore[toSize(t)][toSize(s)] = dW;
            }
        }

        // 2) softmax backward per t
        for (int t{0}; t < T; ++t) {
            float sumDwW{0.0F};
            for (int s{0}; s <= t; ++s) {
                sumDwW += dScore[toSize(t)][toSize(s)] * sc.attnWeight(t, s, h);
            }

            for (int s{0}; s <= t; ++s) {
                const float w{sc.attnWeight(t, s, h)};
                const float dW{dScore[toSize(t)][toSize(s)]};
                dScore[toSize(t)][toSize(s)] = w * (dW - sumDwW);
            }
        }

        // 3) dQ, dK from scores
        for (int t{0}; t < T; ++t) {
            const float* qT{&sc.q[toSize(t)][toSize(hs)]};
            float* dQT{&b.dQ[toSize(t)][toSize(hs)]};

            for (int s{0}; s <= t; ++s) {
                const float ds{dScore[toSize(t)][toSize(s)] * invSqrtHd};
                const float* kS{&sc.k[toSize(s)][toSize(hs)]};
                float* dKS{&b.dK[toSize(s)][toSize(hs)]};

                for (int j{0}; j < headDim; ++j) {
                    dQT[j] += ds * kS[j];
                    dKS[j] += ds * qT[j];
                }
            }
        }
    }
}

// ----------------------------- training (single layer) -----------------------------
// Returns mean loss. Computes forward + backward + accumulates grads.
// Caller is responsible for calling adamStep() afterward.
float trainStep(Model& model,
                TrainScratch& s,
                const Tokenizer& tok,
                const std::vector<std::string>& docs,
                int step,
                std::mt19937& rng,
                int& outTokenCount) noexcept {
    (void)rng;

    // Tokenize one doc into tokens[0..len-1], with BOS at both ends.
    std::array<int, blockSize + 2> tokens{};
    tokens[0] = tok.bosId;

    const std::string& doc{docs[toSize(step % static_cast<int>(docs.size()))]};

    int len{1};
    for (char c : doc) {
        if (len >= blockSize + 1) {
            break;
        }

        auto idExp{tok.encodeChar(c)};
        if (!idExp) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        tokens[toSize(len)] = *idExp;
        len += 1;
    }

    tokens[toSize(len)] = tok.bosId;
    len += 1;

    const int T{std::min(blockSize, len - 1)};
    outTokenCount = T;

    // Clear grads
    model.zeroGrads();

    // ---------------- forward ----------------
    // xEmbSum, x0
    for (int t{0}; t < T; ++t) {
        const int tokenId{tokens[toSize(t)]};

        const uint32_t wteRow{model.off.wte + toU32(tokenId * nEmbd)};
        const uint32_t wpeRow{model.off.wpe + toU32(t * nEmbd)};

        for (int i{0}; i < nEmbd; ++i) {
            const float te{model.params[toSize(static_cast<int>(wteRow + toU32(i)))]};
            const float pe{model.params[toSize(static_cast<int>(wpeRow + toU32(i)))]};
            s.xEmbSum[toSize(t)][toSize(i)] = te + pe;
        }

        s.inv0[toSize(t)] = rmsnormForward(s.xEmbSum[toSize(t)].data(), s.x0[toSize(t)].data(), nEmbd);
    }

    // Layer 0: norm1 -> qkv
    for (int t{0}; t < T; ++t) {
        s.inv1[toSize(t)] = rmsnormForward(s.x0[toSize(t)].data(), s.x1[toSize(t)].data(), nEmbd);

        const float* x{ s.x1[toSize(t)].data() };

        linearForward(&model.params[toSize(static_cast<int>(model.off.attnWq[0]))], x, s.q[toSize(t)].data(), nEmbd, nEmbd);
        linearForward(&model.params[toSize(static_cast<int>(model.off.attnWk[0]))], x, s.k[toSize(t)].data(), nEmbd, nEmbd);
        linearForward(&model.params[toSize(static_cast<int>(model.off.attnWv[0]))], x, s.v[toSize(t)].data(), nEmbd, nEmbd);
    }

    // Attention
    attentionForward(s, s, T);

    // attnProj = Wo * attnConcat, x2 = x0 + attnProj
    for (int t{0}; t < T; ++t) {
        linearForward(&model.params[toSize(static_cast<int>(model.off.attnWo[0]))],
                      s.attnConcat[toSize(t)].data(),
                      s.attnProj[toSize(t)].data(),
                      nEmbd,
                      nEmbd);

        addVec(s.x0[toSize(t)].data(), s.attnProj[toSize(t)].data(), s.x2[toSize(t)].data(), nEmbd);
    }

    // norm2 -> mlp -> xOut
    for (int t{0}; t < T; ++t) {
        s.inv2[toSize(t)] = rmsnormForward(s.x2[toSize(t)].data(), s.x3[toSize(t)].data(), nEmbd);

        // fc1
        linearForward(&model.params[toSize(static_cast<int>(model.off.mlpFc1[0]))],
                      s.x3[toSize(t)].data(),
                      s.fc1Pre[toSize(t)].data(),
                      4 * nEmbd,
                      nEmbd);

        // relu
        for (int i{0}; i < 4 * nEmbd; ++i) {
            const float v{ s.fc1Pre[toSize(t)][toSize(i)] };
            if (v > 0.0F) {
                s.fc1Relu[toSize(t)][toSize(i)] = v;
                s.reluMask[toSize(t)][toSize(i)] = uint8_t{1};
            } else {
                s.fc1Relu[toSize(t)][toSize(i)] = 0.0F;
                s.reluMask[toSize(t)][toSize(i)] = uint8_t{0};
            }
        }

        // fc2
        linearForward(&model.params[toSize(static_cast<int>(model.off.mlpFc2[0]))],
                      s.fc1Relu[toSize(t)].data(),
                      s.fc2[toSize(t)].data(),
                      nEmbd,
                      4 * nEmbd);

        addVec(s.x2[toSize(t)].data(), s.fc2[toSize(t)].data(), s.xOut[toSize(t)].data(), nEmbd);
    }

    // output: logits -> probs -> loss
    float loss{0.0F};

    for (int t{0}; t < T; ++t) {
        const float* x{ s.xOut[toSize(t)].data() };
        const float* wLm{ &model.params[toSize(static_cast<int>(model.off.lmHead))] };

        for (int o{0}; o < model.vocabSize; ++o) {
            const float* row{wLm + o * nEmbd};
            s.logitsTmp[toSize(o)] = dot(row, x, nEmbd);
        }

        const int targetId{tokens[toSize(t + 1)]};
        float* probs{ s.probsRow(t, model.vocabSize) };
        loss += softmaxCrossEntropyForward(s.logitsTmp.data(), probs, model.vocabSize, targetId);
    }

    const float invT{1.0F / float(T)};
    const float lossMean{loss * invT};

    // ---------------- backward ----------------
    // clear backward buffers
    for (int t{0}; t < T; ++t) {
        setZero(s.dXOut[toSize(t)].data(), nEmbd);
        setZero(s.dX2[toSize(t)].data(), nEmbd);
        setZero(s.dAttnProj[toSize(t)].data(), nEmbd);
        setZero(s.dAttnConcat[toSize(t)].data(), nEmbd);
        setZero(s.dX0[toSize(t)].data(), nEmbd);
        setZero(s.dX1[toSize(t)].data(), nEmbd);
        setZero(s.dX3[toSize(t)].data(), nEmbd);
        setZero(s.dFc2[toSize(t)].data(), nEmbd);
        setZero(s.dFc1Relu[toSize(t)].data(), 4 * nEmbd);
        setZero(s.dFc1Pre[toSize(t)].data(), 4 * nEmbd);
    }

    // 1) output layer backward, accumulate dXOut
    const float* wLm{ &model.params[toSize(static_cast<int>(model.off.lmHead))] };
    float* dWLm{ &model.grads[toSize(static_cast<int>(model.off.lmHead))] };

    for (int t{0}; t < T; ++t) {
        const int targetId{tokens[toSize(t + 1)]};
        const float scale{invT};

        const float* probs{ s.probsRow(t, model.vocabSize) };
        const float* xOut{ s.xOut[toSize(t)].data() };
        float* dXOut{ s.dXOut[toSize(t)].data() };

        for (int i{0}; i < nEmbd; ++i) {
            dXOut[i] = 0.0F;
        }

        for (int o{0}; o < model.vocabSize; ++o) {
            float dlog{probs[toSize(o)]};
            if (o == targetId) {
                dlog -= 1.0F;
            }
            dlog *= scale;

            const float* row{wLm + o * nEmbd};
            float* dRow{dWLm + o * nEmbd};

            for (int i{0}; i < nEmbd; ++i) {
                dRow[i] += dlog * xOut[i];
                dXOut[i] += row[i] * dlog;
            }
        }
    }

    // 2) back through xOut = x2 + fc2
    for (int t{0}; t < T; ++t) {
        addInPlace(s.dX2[toSize(t)].data(), s.dXOut[toSize(t)].data(), nEmbd);

        for (int i{0}; i < nEmbd; ++i) {
            s.dFc2[toSize(t)][toSize(i)] = s.dXOut[toSize(t)][toSize(i)];
        }
    }

    // 3) fc2 backward: fc2 = W2 * relu
    {
        const float* w2{ &model.params[toSize(static_cast<int>(model.off.mlpFc2[0]))] };
        float* dW2{ &model.grads[toSize(static_cast<int>(model.off.mlpFc2[0]))] };

        for (int t{0}; t < T; ++t) {
            float dRelu[4 * nEmbd]{};
            for (int i{0}; i < 4 * nEmbd; ++i) {
                dRelu[i] = 0.0F;
            }

            for (int o{0}; o < nEmbd; ++o) {
                const float dyo{s.dFc2[toSize(t)][toSize(o)]};
                const float* row{w2 + o * (4 * nEmbd)};
                float* dRow{dW2 + o * (4 * nEmbd)};

                for (int i{0}; i < 4 * nEmbd; ++i) {
                    dRow[i] += dyo * s.fc1Relu[toSize(t)][toSize(i)];
                    dRelu[i] += row[i] * dyo;
                }
            }

            for (int i{0}; i < 4 * nEmbd; ++i) {
                s.dFc1Pre[toSize(t)][toSize(i)] = (s.reluMask[toSize(t)][toSize(i)] != uint8_t{0}) ? dRelu[i] : 0.0F;
            }
        }
    }

    // 4) fc1 backward: fc1Pre = W1 * x3, accumulate dX3
    {
        const float* w1{ &model.params[toSize(static_cast<int>(model.off.mlpFc1[0]))] };
        float* dW1{ &model.grads[toSize(static_cast<int>(model.off.mlpFc1[0]))] };

        for (int t{0}; t < T; ++t) {
            float* dX3{ s.dX3[toSize(t)].data() };
            for (int i{0}; i < nEmbd; ++i) {
                dX3[i] = 0.0F;
            }

            linearBackwardAcc(w1,
                              s.x3[toSize(t)].data(),
                              s.dFc1Pre[toSize(t)].data(),
                              dW1,
                              dX3,
                              4 * nEmbd,
                              nEmbd);
        }
    }

    // 5) rmsnorm2 backward: x3 = rmsnorm(x2)
    for (int t{0}; t < T; ++t) {
        rmsnormBackward(s.dX3[toSize(t)].data(),
                        s.x2[toSize(t)].data(),
                        s.inv2[toSize(t)],
                        s.dX2[toSize(t)].data(),
                        nEmbd);
    }

    // 6) x2 = x0 + attnProj
    for (int t{0}; t < T; ++t) {
        addInPlace(s.dX0[toSize(t)].data(), s.dX2[toSize(t)].data(), nEmbd);
        addInPlace(s.dAttnProj[toSize(t)].data(), s.dX2[toSize(t)].data(), nEmbd);
    }

    // 7) Wo backward
    {
        const float* wO{ &model.params[toSize(static_cast<int>(model.off.attnWo[0]))] };
        float* dWO{ &model.grads[toSize(static_cast<int>(model.off.attnWo[0]))] };

        for (int t{0}; t < T; ++t) {
            float* dAttnConcat{ s.dAttnConcat[toSize(t)].data() };
            for (int i{0}; i < nEmbd; ++i) {
                dAttnConcat[i] = 0.0F;
            }

            linearBackwardAcc(wO,
                              s.attnConcat[toSize(t)].data(),
                              s.dAttnProj[toSize(t)].data(),
                              dWO,
                              dAttnConcat,
                              nEmbd,
                              nEmbd);
        }
    }

    // 8) attention backward -> dQ,dK,dV
    attentionBackward(s, s, T);

    // 9) qkv linear backward
    for (int t{0}; t < T; ++t) {
        setZero(s.dX1[toSize(t)].data(), nEmbd);
    }

    {
        const float* wQ{ &model.params[toSize(static_cast<int>(model.off.attnWq[0]))] };
        float* dWQ{ &model.grads[toSize(static_cast<int>(model.off.attnWq[0]))] };

        const float* wK{ &model.params[toSize(static_cast<int>(model.off.attnWk[0]))] };
        float* dWK{ &model.grads[toSize(static_cast<int>(model.off.attnWk[0]))] };

        const float* wV{ &model.params[toSize(static_cast<int>(model.off.attnWv[0]))] };
        float* dWV{ &model.grads[toSize(static_cast<int>(model.off.attnWv[0]))] };

        for (int t{0}; t < T; ++t) {
            linearBackwardAcc(wQ, s.x1[toSize(t)].data(), s.dQ[toSize(t)].data(), dWQ, s.dX1[toSize(t)].data(), nEmbd, nEmbd);
            linearBackwardAcc(wK, s.x1[toSize(t)].data(), s.dK[toSize(t)].data(), dWK, s.dX1[toSize(t)].data(), nEmbd, nEmbd);
            linearBackwardAcc(wV, s.x1[toSize(t)].data(), s.dV[toSize(t)].data(), dWV, s.dX1[toSize(t)].data(), nEmbd, nEmbd);
        }
    }

    // 10) rmsnorm1 backward
    for (int t{0}; t < T; ++t) {
        rmsnormBackward(s.dX1[toSize(t)].data(),
                        s.x0[toSize(t)].data(),
                        s.inv1[toSize(t)],
                        s.dX0[toSize(t)].data(),
                        nEmbd);
    }

    // 11) rmsnorm0 backward -> scatter to embeddings
    for (int t{0}; t < T; ++t) {
        float dXEmbSum[nEmbd]{};
        for (int i{0}; i < nEmbd; ++i) {
            dXEmbSum[i] = 0.0F;
        }

        rmsnormBackward(s.dX0[toSize(t)].data(),
                        s.xEmbSum[toSize(t)].data(),
                        s.inv0[toSize(t)],
                        dXEmbSum,
                        nEmbd);

        const int tokenId{tokens[toSize(t)]};
        const uint32_t wteRow{model.off.wte + toU32(tokenId * nEmbd)};
        const uint32_t wpeRow{model.off.wpe + toU32(t * nEmbd)};

        for (int i{0}; i < nEmbd; ++i) {
            const uint32_t teIdx{wteRow + toU32(i)};
            const uint32_t peIdx{wpeRow + toU32(i)};
            model.grads[toSize(static_cast<int>(teIdx))] += dXEmbSum[i];
            model.grads[toSize(static_cast<int>(peIdx))] += dXEmbSum[i];
        }
    }

    // FIX: Do NOT call adamStep here. The caller handles it with the correct schedule.
    return lossMean;
}

// ----------------------------- sampling (inference) -----------------------------
int sampleCategorical(const std::vector<float>& probs, std::mt19937& rng) noexcept {
    float sum{0.0F};
    for (float p : probs) {
        sum += p;
    }

    if (!(sum > 0.0F)) {
        std::uniform_int_distribution<int> uid{0, static_cast<int>(probs.size()) - 1};
        return uid(rng);
    }

    std::uniform_real_distribution<float> urd{0.0F, sum};
    const float r{urd(rng)};

    float cdf{0.0F};
    for (int i{0}; i < static_cast<int>(probs.size()); ++i) {
        cdf += probs[toSize(i)];
        if (r <= cdf) {
            return i;
        }
    }

    return static_cast<int>(probs.size()) - 1;
}

struct InferScratch {
    std::array<float, nEmbd> xEmbSum{};
    std::array<float, nEmbd> x0{};
    std::array<float, nEmbd> x1{};
    std::array<float, nEmbd> q{};
    std::array<float, nEmbd> k{};
    std::array<float, nEmbd> v{};
    std::array<float, nEmbd> attnConcat{};
    std::array<float, nEmbd> attnProj{};
    std::array<float, nEmbd> x2{};
    std::array<float, nEmbd> x3{};
    std::array<float, 4 * nEmbd> fc1{};
    std::array<float, nEmbd> fc2{};
    std::array<float, nEmbd> xOut{};

    std::vector<float> logits{};
    std::vector<float> probs{};

    Expected<void> init(int vocabSize) noexcept {
        logits.assign(toSize(vocabSize), 0.0F);
        probs.assign(toSize(vocabSize), 0.0F);
        return {};
    }
};

void runInference(const Model& model,
                  const Tokenizer& tok,
                  int numSamples,
                  float temperature,
                  std::mt19937& rng) {
    std::cout << "\n--- inference (new, hallucinated names) ---\n";

    InferScratch s{};
    (void)s.init(model.vocabSize);

    std::array<std::array<std::array<float, nEmbd>, blockSize>, nLayer> kCache{};
    std::array<std::array<std::array<float, nEmbd>, blockSize>, nLayer> vCache{};
    std::array<int, nLayer> cacheLen{};

    const float invSqrtHd{1.0F / std::sqrt(float(headDim))};

    uint64_t totalGeneratedChars{0u};

    for (int sampleIdx{0}; sampleIdx < numSamples; ++sampleIdx) {
        for (int li{0}; li < nLayer; ++li) {
            cacheLen[toSize(li)] = 0;
        }

        int tokenId{tok.bosId};
        std::array<char, blockSize + 1> out{};
        int outLen{0};

        for (int posId{0}; posId < blockSize; ++posId) {
            // embedding + norm0
            const uint32_t wteRow{model.off.wte + toU32(tokenId * nEmbd)};
            const uint32_t wpeRow{model.off.wpe + toU32(posId * nEmbd)};
            for (int i{0}; i < nEmbd; ++i) {
                const float te{model.params[toSize(static_cast<int>(wteRow + toU32(i)))]};
                const float pe{model.params[toSize(static_cast<int>(wpeRow + toU32(i)))]};
                s.xEmbSum[toSize(i)] = te + pe;
            }
            (void)rmsnormForward(s.xEmbSum.data(), s.x0.data(), nEmbd);

            // norm1 -> qkv
            (void)rmsnormForward(s.x0.data(), s.x1.data(), nEmbd);

            linearForward(&model.params[toSize(static_cast<int>(model.off.attnWq[0]))], s.x1.data(), s.q.data(), nEmbd, nEmbd);
            linearForward(&model.params[toSize(static_cast<int>(model.off.attnWk[0]))], s.x1.data(), s.k.data(), nEmbd, nEmbd);
            linearForward(&model.params[toSize(static_cast<int>(model.off.attnWv[0]))], s.x1.data(), s.v.data(), nEmbd, nEmbd);

            const int t{cacheLen[0]};
            for (int i{0}; i < nEmbd; ++i) {
                kCache[0][toSize(t)][toSize(i)] = s.k[toSize(i)];
                vCache[0][toSize(t)][toSize(i)] = s.v[toSize(i)];
            }
            cacheLen[0] = t + 1;

            // attention output
            for (int i{0}; i < nEmbd; ++i) {
                s.attnConcat[toSize(i)] = 0.0F;
            }

            for (int h{0}; h < nHead; ++h) {
                const int hs{h * headDim};

                float maxScore{-std::numeric_limits<float>::infinity()};
                std::array<float, blockSize> score{};
                for (int ss{0}; ss < cacheLen[0]; ++ss) {
                    float d{0.0F};
                    for (int j{0}; j < headDim; ++j) {
                        d += s.q[toSize(hs + j)] * kCache[0][toSize(ss)][toSize(hs + j)];
                    }
                    d *= invSqrtHd;
                    score[toSize(ss)] = d;
                    maxScore = std::max(maxScore, d);
                }

                std::array<float, blockSize> w{};
                float sumExp{0.0F};
                for (int ss{0}; ss < cacheLen[0]; ++ss) {
                    const float e{std::exp(score[toSize(ss)] - maxScore)};
                    w[toSize(ss)] = e;
                    sumExp += e;
                }
                const float invSum{1.0F / sumExp};
                for (int ss{0}; ss < cacheLen[0]; ++ss) {
                    w[toSize(ss)] *= invSum;
                }

                for (int j{0}; j < headDim; ++j) {
                    float acc{0.0F};
                    for (int ss{0}; ss < cacheLen[0]; ++ss) {
                        acc += w[toSize(ss)] * vCache[0][toSize(ss)][toSize(hs + j)];
                    }
                    s.attnConcat[toSize(hs + j)] = acc;
                }
            }

            // proj + residual
            linearForward(&model.params[toSize(static_cast<int>(model.off.attnWo[0]))], s.attnConcat.data(), s.attnProj.data(), nEmbd, nEmbd);
            addVec(s.x0.data(), s.attnProj.data(), s.x2.data(), nEmbd);

            // norm2 + mlp + residual
            (void)rmsnormForward(s.x2.data(), s.x3.data(), nEmbd);

            linearForward(&model.params[toSize(static_cast<int>(model.off.mlpFc1[0]))], s.x3.data(), s.fc1.data(), 4 * nEmbd, nEmbd);
            for (int i{0}; i < 4 * nEmbd; ++i) {
                s.fc1[toSize(i)] = (s.fc1[toSize(i)] > 0.0F) ? s.fc1[toSize(i)] : 0.0F;
            }

            linearForward(&model.params[toSize(static_cast<int>(model.off.mlpFc2[0]))], s.fc1.data(), s.fc2.data(), nEmbd, 4 * nEmbd);
            addVec(s.x2.data(), s.fc2.data(), s.xOut.data(), nEmbd);

            // logits
            const float* wLm{ &model.params[toSize(static_cast<int>(model.off.lmHead))] };
            for (int o{0}; o < model.vocabSize; ++o) {
                const float* row{wLm + o * nEmbd};
                s.logits[toSize(o)] = dot(row, s.xOut.data(), nEmbd);
            }

            // sample
            float maxVal{-std::numeric_limits<float>::infinity()};
            for (int i{0}; i < model.vocabSize; ++i) {
                maxVal = std::max(maxVal, s.logits[toSize(i)] / temperature);
            }

            float sumExp{0.0F};
            for (int i{0}; i < model.vocabSize; ++i) {
                const float e{std::exp((s.logits[toSize(i)] / temperature) - maxVal)};
                s.probs[toSize(i)] = e;
                sumExp += e;
            }

            const float invSum{1.0F / sumExp};
            for (int i{0}; i < model.vocabSize; ++i) {
                s.probs[toSize(i)] *= invSum;
            }

            // FIX: Suppress BOS at position 0 to prevent blank samples.
            // The first generated character must be a real character, not EOS.
            if (posId == 0) {
                s.probs[toSize(tok.bosId)] = 0.0F;
                float reSum{0.0F};
                for (int i{0}; i < model.vocabSize; ++i) {
                    reSum += s.probs[toSize(i)];
                }
                if (reSum > 0.0F) {
                    const float reInv{1.0F / reSum};
                    for (int i{0}; i < model.vocabSize; ++i) {
                        s.probs[toSize(i)] *= reInv;
                    }
                }
            }

            tokenId = sampleCategorical(s.probs, rng);
            if (tokenId == tok.bosId) {
                break;
            }

            out[toSize(outLen)] = tok.decodeId(tokenId);
            outLen += 1;
        }

        out[toSize(outLen)] = '\0';
        totalGeneratedChars += static_cast<uint64_t>(outLen);

        std::cout << "sample " << std::setw(2) << (sampleIdx + 1) << ": " << out.data() << "\n";
    }

    std::cout << "\n";
    std::cout << std::left << std::setw(34) << "generated chars (total)"
              << std::right << std::setw(12) << totalGeneratedChars << "\n";
}

// ----------------------------- main -----------------------------
struct Args {
    std::string_view datasetPath{"input.txt"};
    int steps{defaultSteps};
    int samples{defaultSamples};
    int logEvery{defaultLogEvery};
    float temperature{defaultTemperature};
};

Args parseArgs(int argc, char** argv) noexcept {
    Args a{};

    for (int i{1}; i < argc; ++i) {
        const std::string_view arg{(argv[i] != nullptr) ? std::string_view{argv[i]} : std::string_view{}};

        if (arg == "--steps" && (i + 1) < argc) {
            a.steps = std::max(1, parseInt(std::string_view{argv[i + 1]}, a.steps));
            i += 1;
        } else if (arg == "--samples" && (i + 1) < argc) {
            a.samples = std::max(1, parseInt(std::string_view{argv[i + 1]}, a.samples));
            i += 1;
        } else if (arg == "--log-every" && (i + 1) < argc) {
            a.logEvery = std::max(1, parseInt(std::string_view{argv[i + 1]}, a.logEvery));
            i += 1;
        } else if (arg == "--temperature" && (i + 1) < argc) {
            a.temperature = std::max(1e-6F, parseFloat(std::string_view{argv[i + 1]}, a.temperature));
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

        const auto tProgramStart{Clock::now()};
        const Args args{parseArgs(argc, argv)};

        std::mt19937 rng{42u};

        // init
        const auto tInitStart{Clock::now()};

        auto docsExp{readDocs(args.datasetPath)};
        if (!docsExp) {
            std::cerr << "error: " << docsExp.error().message << " (" << args.datasetPath << ")\n";
            return 1;
        }
        auto docs{std::move(*docsExp)};
        std::shuffle(docs.begin(), docs.end(), rng);

        std::cout << "num docs: " << docs.size() << "\n";

        auto tokExp{Tokenizer::build(docs)};
        if (!tokExp) {
            std::cerr << "error: tokenizer build failed\n";
            return 1;
        }
        const Tokenizer tok{*tokExp};

        std::cout << "vocab size: " << tok.vocabSize << "\n";

        Model model{};
        auto mi{model.init(tok.vocabSize, tok.bosId)};
        if (!mi) {
            std::cerr << "error: model init failed\n";
            return 1;
        }
        model.initParams(rng);

        std::cout << "num params: " << model.params.size() << "\n";

        TrainScratch scratch{};
        auto si{scratch.init(model.vocabSize)};
        if (!si) {
            std::cerr << "error: scratch init failed\n";
            return 1;
        }

        const auto tInitEnd{Clock::now()};

        // training
        const auto tTrainStart{Clock::now()};

        float lossFirst{0.0F};
        float lossLast{0.0F};
        float lossMin{std::numeric_limits<float>::infinity()};
        float lossSum{0.0F};
        int lastTokenCount{0};

        for (int step{0}; step < args.steps; ++step) {
            // FIX: trainStep computes forward/backward and accumulates grads.
            // Adam update happens once here with the correct schedule.
            const float loss{trainStep(model, scratch, tok, docs, step, rng, lastTokenCount)};
            model.adamStep(step, args.steps);

            if (step == 0) {
                lossFirst = loss;
            }
            lossLast = loss;
            lossMin = std::min(lossMin, loss);
            lossSum += loss;

            const bool shouldLog{((step + 1) % args.logEvery) == 0 || (step + 1) == 1 || (step + 1) == args.steps};
            if (shouldLog) {
                std::cout << "step " << std::setw(4) << (step + 1) << " / " << std::setw(4) << args.steps
                          << " | loss " << std::fixed << std::setprecision(4) << loss << "\n";
            }
        }

        const auto tTrainEnd{Clock::now()};

        // inference
        const auto tInferStart{Clock::now()};
        runInference(model, tok, args.samples, args.temperature, rng);
        const auto tInferEnd{Clock::now()};

        const auto tProgramEnd{Clock::now()};

        // summary
        const auto initDur{tInitEnd - tInitStart};
        const auto trainDur{tTrainEnd - tTrainStart};
        const auto inferDur{tInferEnd - tInferStart};
        const auto totalDur{tProgramEnd - tProgramStart};

        const float lossAvg{lossSum / float(args.steps)};

        const auto printKey = [](std::string_view key) {
            std::cout << std::left << std::setw(34) << key;
        };

        const auto printMs = [&](std::string_view key, Clock::duration d) {
            printKey(key);
            std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(2) << toMilliseconds(d) << " ms\n";
        };

        const auto printUs = [&](std::string_view key, double us) {
            printKey(key);
            std::cout << std::right << std::setw(12) << std::fixed << std::setprecision(2) << us << " us\n";
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

        printUs("training per step", toMicroseconds(trainDur) / double(args.steps));
        printUs("inference per sample", toMicroseconds(inferDur) / double(args.samples));

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
