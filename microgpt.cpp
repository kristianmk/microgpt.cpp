// Written by K. M. Knausgård 2026-02-14, based on https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
//
// C++23, std-only.
//
// Goals:
// - No heap allocations after initialization.
// - No exceptions after initialization (init exceptions caught in main).
// - Predictable error handling using std::expected for init/IO.
// - Always use braces for control flow.
// - Autograd arena capacity calculated exactly (no slack) from constexpr formulas + runtime vocabSize.
// - Brief end-of-run timing + training statistics (init vs training vs inference), aligned output.
// - Prefer {} initialization to reduce narrowing/implicit conversions.
// - Optional exact reproduction mode: --karpathy-py-compat (CPython RNG + Python topo-backward order).

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
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

constexpr double initStd{0.08};
constexpr double rmsEps{1e-5};

constexpr int numSteps{1000};

constexpr double learningRate{0.01};
constexpr double beta1{0.85};
constexpr double beta2{0.99};
constexpr double epsAdam{1e-8};

constexpr double temperature{0.5};

// ----------------------------- run mode -----------------------------
enum class RunMode : uint8_t {
    fast,
    pythonCompat
};

// ----------------------------- error handling -----------------------------
enum class ErrorCode : uint8_t {
    fileOpenFailed,
    fileReadFailed,
    emptyDataset,
    invalidChar,
    arenaTooSmall
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

// ----------------------------- Python-compatible RNG -----------------------------
// Implements CPython random.Random core behavior for:
// - seed(int) using init_by_array (Mersenne Twister 19937)
// - random() (53-bit float construction)
// - gauss() (Box-Muller with cached spare)
// - shuffle() using randbelow/getrandbits rejection
// - choices(population, weights) via random()*total and bisect_right behavior (r < cum)
class PythonRandom {
public:
    explicit PythonRandom(uint64_t seedValue) noexcept {
        seed(seedValue);
    }

    void seed(uint64_t seedValue) noexcept {
        hasGaussNext_ = false;

        std::array<uint32_t, 2> keyTmp{};
        size_t keyLen{0};

        if (seedValue == 0u) {
            keyTmp[0] = 0u;
            keyLen = 1;
        } else {
            while (seedValue != 0u && keyLen < keyTmp.size()) {
                keyTmp[keyLen] = static_cast<uint32_t>(seedValue & 0xFFFF'FFFFu);
                seedValue >>= 32u;
                keyLen += 1;
            }
        }

        initByArray_(keyTmp.data(), keyLen);
    }

    double random01() noexcept {
        // CPython: a = genrand_uint32() >> 5 (27 bits), b = genrand_uint32() >> 6 (26 bits)
        // return (a*2^26 + b) / 2^53
        const uint32_t a{genrandUint32_() >> 5u};
        const uint32_t b{genrandUint32_() >> 6u};
        const double x{(static_cast<double>(a) * 67108864.0) + static_cast<double>(b)};
        return x / 9007199254740992.0;
    }

    double gauss(double mu, double sigma) noexcept {
        if (hasGaussNext_) {
            hasGaussNext_ = false;
            return mu + gaussNext_ * sigma;
        }

        // CPython random.py uses: x2pi = random()*2*pi; g2rad = sqrt(-2*log(1-random()))
        constexpr double twoPi{6.283185307179586476925286766559};
        const double x2pi{random01() * twoPi};

        // Use 1 - u to avoid log(0)
        const double u{1.0 - random01()};
        const double g2rad{std::sqrt(-2.0 * std::log(u))};

        const double z0{std::cos(x2pi) * g2rad};
        const double z1{std::sin(x2pi) * g2rad};

        gaussNext_ = z1;
        hasGaussNext_ = true;

        return mu + z0 * sigma;
    }

    template <class T>
    void shuffle(std::vector<T>& v) noexcept {
        const size_t n{v.size()};
        if (n <= 1u) {
            return;
        }

        // Fisher-Yates with Python's randbelow(i+1)
        for (size_t i{n - 1u}; i > 0u; --i) {
            const uint32_t j{randBelow_(static_cast<uint32_t>(i + 1u))};
            std::swap(v[i], v[toSize(static_cast<int>(j))]);
        }
    }

    size_t choicesIndex(const std::vector<double>& weights) noexcept {
        // Python choices uses: cum_weights; return bisect(cum_weights, random()*total)
        // bisect is bisect_right => choose first cum_weight > r, i.e. r < cum
        double total{0.0};
        for (double w : weights) {
            total += w;
        }

        const double r{random01() * total};

        double c{0.0};
        for (size_t i{0}; i < weights.size(); ++i) {
            c += weights[i];
            if (r < c) {
                return i;
            }
        }

        return (weights.empty()) ? 0u : (weights.size() - 1u);
    }

private:
    static constexpr uint32_t n_{624u};
    static constexpr uint32_t m_{397u};
    static constexpr uint32_t matrixA_{0x9908B0DFu};
    static constexpr uint32_t upperMask_{0x80000000u};
    static constexpr uint32_t lowerMask_{0x7FFFFFFFu};

    void initGenrand_(uint32_t s) noexcept {
        mt_[0] = s;
        for (uint32_t i{1u}; i < n_; ++i) {
            const uint32_t prev{mt_[i - 1u]};
            mt_[i] = static_cast<uint32_t>(1812433253u * (prev ^ (prev >> 30u)) + i);
        }
        index_ = n_;
    }

    void initByArray_(const uint32_t* key, size_t keyLength) noexcept {
        initGenrand_(19650218u);

        uint32_t i{1u};
        uint32_t j{0u};

        const uint32_t kMax{(n_ > static_cast<uint32_t>(keyLength)) ? n_ : static_cast<uint32_t>(keyLength)};
        for (uint32_t k{0u}; k < kMax; ++k) {
            const uint32_t prev{mt_[i - 1u]};
            const uint32_t x{(prev ^ (prev >> 30u))};
            const uint32_t keyVal{(keyLength > 0u) ? key[j] : 0u};

            mt_[i] = static_cast<uint32_t>((mt_[i] ^ (x * 1664525u)) + keyVal + j);

            i += 1u;
            j += 1u;

            if (i >= n_) {
                mt_[0] = mt_[n_ - 1u];
                i = 1u;
            }
            if (j >= static_cast<uint32_t>(keyLength)) {
                j = 0u;
            }
        }

        for (uint32_t k{0u}; k < (n_ - 1u); ++k) {
            const uint32_t prev{mt_[i - 1u]};
            const uint32_t x{(prev ^ (prev >> 30u))};

            mt_[i] = static_cast<uint32_t>((mt_[i] ^ (x * 1566083941u)) - i);

            i += 1u;
            if (i >= n_) {
                mt_[0] = mt_[n_ - 1u];
                i = 1u;
            }
        }

        mt_[0] = 0x80000000u;
        index_ = n_;
    }

    void twist_() noexcept {
        for (uint32_t i{0u}; i < n_; ++i) {
            const uint32_t y{(mt_[i] & upperMask_) | (mt_[(i + 1u) % n_] & lowerMask_)};
            uint32_t x{mt_[(i + m_) % n_] ^ (y >> 1u)};
            if ((y & 1u) != 0u) {
                x ^= matrixA_;
            }
            mt_[i] = x;
        }
        index_ = 0u;
    }

    uint32_t genrandUint32_() noexcept {
        if (index_ >= n_) {
            twist_();
        }

        uint32_t y{mt_[index_]};
        index_ += 1u;

        // Tempering
        y ^= (y >> 11u);
        y ^= (y << 7u) & 0x9D2C5680u;
        y ^= (y << 15u) & 0xEFC60000u;
        y ^= (y >> 18u);

        return y;
    }

    uint32_t getRandBits_(uint32_t k) noexcept {
        // Only needed for small k here (shuffle), but keep correct for k in [1..32].
        // CPython discards extra high bits by shifting right.
        const uint32_t y{genrandUint32_()};
        return (k >= 32u) ? y : (y >> (32u - k));
    }

    uint32_t randBelow_(uint32_t n) noexcept {
        // Equivalent to Python's _randbelow_with_getrandbits.
        if (n == 0u) {
            return 0u;
        }

        const uint32_t k{static_cast<uint32_t>(std::bit_width(n))};
        uint32_t r{getRandBits_(k)};
        while (r >= n) {
            r = getRandBits_(k);
        }
        return r;
    }

    std::array<uint32_t, n_> mt_{};
    uint32_t index_{n_};

    bool hasGaussNext_{false};
    double gaussNext_{0.0};
};

// ----------------------------- autodiff arena -----------------------------
using Ref = uint32_t;
constexpr Ref refParamBit{0x8000'0000u};

inline Ref makeNodeRef(uint32_t nodeIndex) noexcept { return nodeIndex; }
inline Ref makeParamRef(uint32_t paramIndex) noexcept { return refParamBit | paramIndex; }
inline bool isParamRef(Ref r) noexcept { return (r & refParamBit) != 0u; }
inline uint32_t refIndex(Ref r) noexcept { return r & ~refParamBit; }

struct Node {
    double data{0.0};
    double grad{0.0};

    Ref a{0};
    Ref b{0};
    double dA{0.0};
    double dB{0.0};
    bool hasB{false};
};

class AutogradArena {
public:
    Expected<void> init(size_t capacity, bool enablePythonTopo) noexcept {
        nodes_.assign(capacity, Node{});
        size_ = 0;

        pythonTopoEnabled_ = enablePythonTopo;
        if (pythonTopoEnabled_) {
            visitStamp_.assign(capacity, 0u);
            topo_.clear();
            topo_.reserve(capacity);

            stack_.clear();
            stack_.reserve(capacity);

            stamp_ = 1u;
        } else {
            visitStamp_.clear();
            topo_.clear();
            stack_.clear();
            stamp_ = 1u;
        }

        return {};
    }

    void reset() noexcept {
        size_ = 0;
    }

    uint32_t size() const noexcept {
        return size_;
    }

    size_t capacity() const noexcept {
        return nodes_.size();
    }

    Expected<uint32_t> createLeaf(double v) noexcept {
        auto idx{allocNode_()};
        if (!idx) {
            return std::unexpected(idx.error());
        }

        Node& n{nodes_[*idx]};
        n.data = v;
        n.grad = 0.0;
        n.a = 0;
        n.b = 0;
        n.dA = 0.0;
        n.dB = 0.0;
        n.hasB = false;
        return *idx;
    }

    Expected<uint32_t> createUnary(double v, Ref a, double dA) noexcept {
        auto idx{allocNode_()};
        if (!idx) {
            return std::unexpected(idx.error());
        }

        Node& n{nodes_[*idx]};
        n.data = v;
        n.grad = 0.0;
        n.a = a;
        n.b = 0;
        n.dA = dA;
        n.dB = 0.0;
        n.hasB = false;
        return *idx;
    }

    Expected<uint32_t> createBinary(double v, Ref a, Ref b, double dA, double dB) noexcept {
        auto idx{allocNode_()};
        if (!idx) {
            return std::unexpected(idx.error());
        }

        Node& n{nodes_[*idx]};
        n.data = v;
        n.grad = 0.0;
        n.a = a;
        n.b = b;
        n.dA = dA;
        n.dB = dB;
        n.hasB = true;
        return *idx;
    }

    const Node& node(uint32_t idx) const noexcept {
        assert(idx < size_);
        return nodes_[idx];
    }

    double getData(Ref r, const std::vector<double>& params) const noexcept {
        if (isParamRef(r)) {
            return params[refIndex(r)];
        }
        return nodes_[refIndex(r)].data;
    }

    // Fast mode: reverse creation order (valid reverse-topo if nodes are created after inputs).
    void backwardFromCreationOrder(uint32_t lossNode, std::vector<double>& paramGrads) noexcept {
        nodes_[lossNode].grad = 1.0;

        for (uint32_t i{size_}; i-- > 0u;) {
            const Node& n{nodes_[i]};
            const double g{n.grad};
            if (g == 0.0) {
                continue;
            }

            addGrad_(n.a, n.dA * g, paramGrads);

            if (n.hasB) {
                addGrad_(n.b, n.dB * g, paramGrads);
            }
        }
    }

    // Python-compat: DFS build topo then process reversed(topo).
    void backwardFromPythonTopo(uint32_t lossNode, std::vector<double>& paramGrads) noexcept {
        // Guard: if not enabled, fall back
        if (!pythonTopoEnabled_) {
            backwardFromCreationOrder(lossNode, paramGrads);
            return;
        }

        stamp_ += 1u;
        if (stamp_ == 0u) {
            std::fill(visitStamp_.begin(), visitStamp_.end(), 0u);
            stamp_ = 1u;
        }

        topo_.clear();
        stack_.clear();

        // Frame: node index + phase (0=enter, 1=exit)
        stack_.push_back(Frame{lossNode, 0u});

        while (!stack_.empty()) {
            const Frame f{stack_.back()};
            stack_.pop_back();

            const uint32_t v{f.node};
            if (f.phase == 0u) {
                if (visitStamp_[v] == stamp_) {
                    continue;
                }
                visitStamp_[v] = stamp_;

                // Post-order: push exit, then children (reverse order on stack for correct visit order)
                stack_.push_back(Frame{v, 1u});

                const Node& n{nodes_[v]};

                if (n.hasB && !isParamRef(n.b)) {
                    stack_.push_back(Frame{refIndex(n.b), 0u}); // b visited after a (because pushed first)
                }
                if (!isParamRef(n.a)) {
                    stack_.push_back(Frame{refIndex(n.a), 0u}); // a visited first
                }
            } else {
                topo_.push_back(v);
            }
        }

        nodes_[lossNode].grad = 1.0;

        for (size_t ti{topo_.size()}; ti-- > 0u;) {
            const uint32_t idx{topo_[ti]};
            const Node& n{nodes_[idx]};
            const double g{n.grad};
            if (g == 0.0) {
                continue;
            }

            addGrad_(n.a, n.dA * g, paramGrads);

            if (n.hasB) {
                addGrad_(n.b, n.dB * g, paramGrads);
            }
        }
    }

private:
    struct Frame {
        uint32_t node{0u};
        uint8_t phase{0u};
    };

    Expected<uint32_t> allocNode_() noexcept {
        if (size_ >= nodes_.size()) {
            return std::unexpected(Error{ErrorCode::arenaTooSmall, "autograd arena capacity exceeded"});
        }
        return size_++;
    }

    void addGrad_(Ref r, double g, std::vector<double>& paramGrads) noexcept {
        if (g == 0.0) {
            return;
        }

        if (isParamRef(r)) {
            paramGrads[refIndex(r)] += g;
        } else {
            nodes_[refIndex(r)].grad += g;
        }
    }

    std::vector<Node> nodes_{};
    uint32_t size_{0};

    bool pythonTopoEnabled_{false};
    std::vector<uint32_t> visitStamp_{};
    std::vector<uint32_t> topo_{};
    std::vector<Frame> stack_{};
    uint32_t stamp_{1u};
};

struct Ops {
    AutogradArena& arena;
    const std::vector<double>& params;

    Expected<Ref> leaf(double v) noexcept {
        auto idx{arena.createLeaf(v)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> add(Ref x, Ref y) noexcept {
        const double xv{arena.getData(x, params)};
        const double yv{arena.getData(y, params)};

        auto idx{arena.createBinary(xv + yv, x, y, 1.0, 1.0)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> sub(Ref x, Ref y) noexcept {
        const double xv{arena.getData(x, params)};
        const double yv{arena.getData(y, params)};

        auto idx{arena.createBinary(xv - yv, x, y, 1.0, -1.0)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> mul(Ref x, Ref y) noexcept {
        const double xv{arena.getData(x, params)};
        const double yv{arena.getData(y, params)};

        auto idx{arena.createBinary(xv * yv, x, y, yv, xv)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> neg(Ref x) noexcept {
        const double xv{arena.getData(x, params)};

        auto idx{arena.createUnary(-xv, x, -1.0)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> inv(Ref x) noexcept {
        const double xv{arena.getData(x, params)};
        const double v{1.0 / xv};
        const double d{-1.0 / (xv * xv)};

        auto idx{arena.createUnary(v, x, d)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> div(Ref x, Ref y) noexcept {
        auto invY{inv(y)};
        if (!invY) {
            return std::unexpected(invY.error());
        }
        return mul(x, *invY);
    }

    Expected<Ref> exp(Ref x) noexcept {
        const double xv{arena.getData(x, params)};
        const double v{std::exp(xv)};

        auto idx{arena.createUnary(v, x, v)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> log(Ref x) noexcept {
        const double xv{arena.getData(x, params)};
        const double v{std::log(xv)};

        auto idx{arena.createUnary(v, x, 1.0 / xv)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> relu(Ref x) noexcept {
        const double xv{arena.getData(x, params)};
        const double v{(xv > 0.0) ? xv : 0.0};
        const double d{(xv > 0.0) ? 1.0 : 0.0};

        auto idx{arena.createUnary(v, x, d)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }

    Expected<Ref> powConst(Ref x, double p) noexcept {
        const double xv{arena.getData(x, params)};
        const double v{std::pow(xv, p)};
        const double d{p * std::pow(xv, p - 1.0)};

        auto idx{arena.createUnary(v, x, d)};
        if (!idx) {
            return std::unexpected(idx.error());
        }
        return makeNodeRef(*idx);
    }
};

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

// ----------------------------- exact arena node counting -----------------------------
struct ArenaNodes {
    static constexpr uint32_t leaf{1u};
    static constexpr uint32_t add{1u};
    static constexpr uint32_t sub{1u};
    static constexpr uint32_t mul{1u};
    static constexpr uint32_t neg{1u};
    static constexpr uint32_t inv{1u};
    static constexpr uint32_t div{2u}; // inv + mul
    static constexpr uint32_t exp{1u};
    static constexpr uint32_t log{1u};
    static constexpr uint32_t relu{1u};
    static constexpr uint32_t powConst{1u};

    static constexpr uint32_t linear(uint32_t nOut, uint32_t nIn) {
        return nOut * (leaf + 2u * nIn);
    }

    template <uint32_t N>
    static constexpr uint32_t rmsnorm() {
        return 6u + 3u * N;
    }

    static constexpr uint32_t softmax(uint32_t L) {
        return 2u + 3u * L + div * L;
    }

    static constexpr uint32_t attnOneHead(uint32_t T) {
        const uint32_t logits{T * (leaf + 2u * static_cast<uint32_t>(headDim) + leaf + mul)};
        const uint32_t weights{softmax(T)};
        const uint32_t weightedSum{static_cast<uint32_t>(headDim) * (leaf + 2u * T)};
        return logits + weights + weightedSum;
    }

    static constexpr uint32_t attnBlock(uint32_t T) {
        return rmsnorm<static_cast<uint32_t>(nEmbd)>() +
               3u * linear(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd)) +
               static_cast<uint32_t>(nHead) * attnOneHead(T) +
               linear(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd)) +
               static_cast<uint32_t>(nEmbd) * add;
    }

    static constexpr uint32_t mlpBlock() {
        return rmsnorm<static_cast<uint32_t>(nEmbd)>() +
               linear(static_cast<uint32_t>(4 * nEmbd), static_cast<uint32_t>(nEmbd)) +
               static_cast<uint32_t>(4 * nEmbd) * relu +
               linear(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(4 * nEmbd)) +
               static_cast<uint32_t>(nEmbd) * add;
    }

    static constexpr uint32_t embeddingAndNorm() {
        return static_cast<uint32_t>(nEmbd) * add + rmsnorm<static_cast<uint32_t>(nEmbd)>();
    }

    static constexpr uint32_t lmAndLoss(uint32_t vocabSize) {
        return linear(vocabSize, static_cast<uint32_t>(nEmbd)) + softmax(vocabSize) + log + neg + add;
    }

    static constexpr uint32_t onePosition(uint32_t vocabSize, uint32_t T) {
        return embeddingAndNorm() + static_cast<uint32_t>(nLayer) * (attnBlock(T) + mlpBlock()) + lmAndLoss(vocabSize);
    }

    static constexpr uint32_t trainingStepWorst(uint32_t vocabSize) {
        uint32_t total{0u};
        total += leaf; // lossSum
        total += leaf; // invN
        for (uint32_t pos{0u}; pos < static_cast<uint32_t>(blockSize); ++pos) {
            total += onePosition(vocabSize, pos + 1u);
        }
        total += mul; // lossMean
        return total;
    }
};

// ----------------------------- model + scratch -----------------------------
struct Scratch {
    std::array<Ref, nEmbd> x{};
    std::array<Ref, nEmbd> tmp{};
    std::array<Ref, nEmbd> q{};
    std::array<Ref, nEmbd> k{};
    std::array<Ref, nEmbd> v{};
    std::array<Ref, nEmbd> xAttn{};
    std::array<Ref, 4 * nEmbd> fc1{};

    std::array<Ref, blockSize> attnLogits{};
    std::array<Ref, blockSize> attnWeights{};

    std::vector<Ref> logits{};       // [vocabSize]
    std::vector<Ref> probs{};        // [vocabSize]
    std::vector<double> probsData{}; // inference-only

    Expected<void> init(int vocabSize) noexcept {
        logits.assign(toSize(vocabSize), Ref{0});
        probs.assign(toSize(vocabSize), Ref{0});
        probsData.assign(toSize(vocabSize), 0.0);
        return {};
    }
};

struct Model {
    int vocabSize{0};
    int bosId{0};

    std::vector<double> params{};
    std::vector<double> paramGrads{};
    std::vector<double> m{};
    std::vector<double> v{};

    struct Offsets {
        uint32_t wte{0};
        uint32_t wpe{0};
        uint32_t lmHead{0};

        std::array<uint32_t, nLayer> attnWq{};
        std::array<uint32_t, nLayer> attnWk{};
        std::array<uint32_t, nLayer> attnWv{};
        std::array<uint32_t, nLayer> attnWo{};

        std::array<uint32_t, nLayer> mlpFc1{};
        std::array<uint32_t, nLayer> mlpFc2{};

        uint32_t totalParams{0};
    } off{};

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
            off.attnWq[li] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.attnWk[li] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.attnWv[li] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.attnWo[li] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(nEmbd));

            off.mlpFc1[li] = cursor;
            cursor += mat(static_cast<uint32_t>(4 * nEmbd), static_cast<uint32_t>(nEmbd));

            off.mlpFc2[li] = cursor;
            cursor += mat(static_cast<uint32_t>(nEmbd), static_cast<uint32_t>(4 * nEmbd));
        }

        off.totalParams = cursor;

        params.assign(toSize(static_cast<int>(off.totalParams)), 0.0);
        paramGrads.assign(toSize(static_cast<int>(off.totalParams)), 0.0);
        m.assign(toSize(static_cast<int>(off.totalParams)), 0.0);
        v.assign(toSize(static_cast<int>(off.totalParams)), 0.0);
        return {};
    }

    void initParamsFast(std::mt19937& rng) noexcept {
        std::normal_distribution<double> nd{0.0, initStd};
        for (double& p : params) {
            p = nd(rng);
        }
    }

    void initParamsPython(PythonRandom& rng) noexcept {
        for (double& p : params) {
            p = rng.gauss(0.0, initStd);
        }
    }

    void zeroParamGrads() noexcept {
        std::fill(paramGrads.begin(), paramGrads.end(), 0.0);
    }

    void adamStep(int step) noexcept {
        const double lrT{learningRate * (1.0 - double(step) / double(numSteps))};
        const double b1t{std::pow(beta1, double(step + 1))};
        const double b2t{std::pow(beta2, double(step + 1))};

        for (size_t i{0}; i < params.size(); ++i) {
            const double g{paramGrads[i]};
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g);

            const double mHat{m[i] / (1.0 - b1t)};
            const double vHat{v[i] / (1.0 - b2t)};

            params[i] -= lrT * mHat / (std::sqrt(vHat) + epsAdam);
        }

        zeroParamGrads();
    }

    using KvCache = std::array<std::array<std::array<Ref, nEmbd>, blockSize>, nLayer>;

    template <size_t NIn, size_t NOut>
    Expected<void> linear(Ops& ops, uint32_t wBase, const std::array<Ref, NIn>& x, std::array<Ref, NOut>& y) const noexcept {
        for (size_t o{0}; o < NOut; ++o) {
            auto acc{ops.leaf(0.0)};
            if (!acc) {
                return std::unexpected(acc.error());
            }

            const uint32_t rowBase{wBase + static_cast<uint32_t>(o * NIn)};

            for (size_t i{0}; i < NIn; ++i) {
                const Ref w{makeParamRef(rowBase + static_cast<uint32_t>(i))};

                auto prod{ops.mul(w, x[i])};
                if (!prod) {
                    return std::unexpected(prod.error());
                }

                auto sum{ops.add(*acc, *prod)};
                if (!sum) {
                    return std::unexpected(sum.error());
                }

                acc = sum;
            }

            y[o] = *acc;
        }

        return {};
    }

    Expected<void> linearVocab(Ops& ops, uint32_t wBase, const std::array<Ref, nEmbd>& x, std::vector<Ref>& y) const noexcept {
        for (int o{0}; o < vocabSize; ++o) {
            auto acc{ops.leaf(0.0)};
            if (!acc) {
                return std::unexpected(acc.error());
            }

            const uint32_t rowBase{wBase + toU32(o * nEmbd)};

            for (int i{0}; i < nEmbd; ++i) {
                const Ref w{makeParamRef(rowBase + toU32(i))};

                auto prod{ops.mul(w, x[toSize(i)])};
                if (!prod) {
                    return std::unexpected(prod.error());
                }

                auto sum{ops.add(*acc, *prod)};
                if (!sum) {
                    return std::unexpected(sum.error());
                }

                acc = sum;
            }

            y[toSize(o)] = *acc;
        }

        return {};
    }

    template <size_t N>
    Expected<void> rmsnorm(Ops& ops, const std::array<Ref, N>& x, std::array<Ref, N>& y) const noexcept {
        auto ms{ops.leaf(0.0)};
        if (!ms) {
            return std::unexpected(ms.error());
        }

        auto invNRef{ops.leaf(1.0 / double(N))};
        if (!invNRef) {
            return std::unexpected(invNRef.error());
        }

        for (size_t i{0}; i < N; ++i) {
            auto xi2{ops.mul(x[i], x[i])};
            if (!xi2) {
                return std::unexpected(xi2.error());
            }

            auto sum{ops.add(*ms, *xi2)};
            if (!sum) {
                return std::unexpected(sum.error());
            }

            ms = sum;
        }

        auto msScaled{ops.mul(*ms, *invNRef)};
        if (!msScaled) {
            return std::unexpected(msScaled.error());
        }

        auto epsRef{ops.leaf(rmsEps)};
        if (!epsRef) {
            return std::unexpected(epsRef.error());
        }

        auto msEps{ops.add(*msScaled, *epsRef)};
        if (!msEps) {
            return std::unexpected(msEps.error());
        }

        auto scale{ops.powConst(*msEps, -0.5)};
        if (!scale) {
            return std::unexpected(scale.error());
        }

        for (size_t i{0}; i < N; ++i) {
            auto yi{ops.mul(x[i], *scale)};
            if (!yi) {
                return std::unexpected(yi.error());
            }
            y[i] = *yi;
        }

        return {};
    }

    Expected<void> softmax(Ops& ops, const std::vector<Ref>& logits, std::vector<Ref>& probs) const noexcept {
        double maxVal{-std::numeric_limits<double>::infinity()};

        for (int i{0}; i < vocabSize; ++i) {
            const double v0{ops.arena.getData(logits[toSize(i)], ops.params)};
            maxVal = std::max(maxVal, v0);
        }

        auto maxConst{ops.leaf(maxVal)};
        if (!maxConst) {
            return std::unexpected(maxConst.error());
        }

        auto sumExp{ops.leaf(0.0)};
        if (!sumExp) {
            return std::unexpected(sumExp.error());
        }

        for (int i{0}; i < vocabSize; ++i) {
            auto shifted{ops.sub(logits[toSize(i)], *maxConst)};
            if (!shifted) {
                return std::unexpected(shifted.error());
            }

            auto e{ops.exp(*shifted)};
            if (!e) {
                return std::unexpected(e.error());
            }

            probs[toSize(i)] = *e;

            auto sum{ops.add(*sumExp, *e)};
            if (!sum) {
                return std::unexpected(sum.error());
            }

            sumExp = sum;
        }

        for (int i{0}; i < vocabSize; ++i) {
            auto divv{ops.div(probs[toSize(i)], *sumExp)};
            if (!divv) {
                return std::unexpected(divv.error());
            }
            probs[toSize(i)] = *divv;
        }

        return {};
    }

    Expected<void> gptStep(Ops& ops,
                           int tokenId,
                           int posId,
                           KvCache& keyCache,
                           KvCache& valueCache,
                           std::array<int, nLayer>& cacheLen,
                           Scratch& sc) const noexcept {
        const uint32_t wteRow{off.wte + toU32(tokenId * nEmbd)};
        const uint32_t wpeRow{off.wpe + toU32(posId * nEmbd)};

        for (int i{0}; i < nEmbd; ++i) {
            const Ref t{makeParamRef(wteRow + toU32(i))};
            const Ref p{makeParamRef(wpeRow + toU32(i))};

            auto sum{ops.add(t, p)};
            if (!sum) {
                return std::unexpected(sum.error());
            }
            sc.x[toSize(i)] = *sum;
        }

        auto rn0{rmsnorm(ops, sc.x, sc.tmp)};
        if (!rn0) {
            return std::unexpected(rn0.error());
        }
        sc.x = sc.tmp;

        for (int li{0}; li < nLayer; ++li) {
            const auto xResidual{sc.x};

            auto rn1{rmsnorm(ops, sc.x, sc.tmp)};
            if (!rn1) {
                return std::unexpected(rn1.error());
            }
            sc.x = sc.tmp;

            auto lq{linear(ops, off.attnWq[li], sc.x, sc.q)};
            if (!lq) {
                return std::unexpected(lq.error());
            }

            auto lk{linear(ops, off.attnWk[li], sc.x, sc.k)};
            if (!lk) {
                return std::unexpected(lk.error());
            }

            auto lv{linear(ops, off.attnWv[li], sc.x, sc.v)};
            if (!lv) {
                return std::unexpected(lv.error());
            }

            const int t{cacheLen[toSize(li)]};
            if (t >= blockSize) {
                return std::unexpected(Error{ErrorCode::arenaTooSmall, "cacheLen exceeded blockSize"});
            }

            for (int i{0}; i < nEmbd; ++i) {
                keyCache[toSize(li)][toSize(t)][toSize(i)] = sc.k[toSize(i)];
                valueCache[toSize(li)][toSize(t)][toSize(i)] = sc.v[toSize(i)];
            }
            cacheLen[toSize(li)] = t + 1;

            for (int h{0}; h < nHead; ++h) {
                const int hs{h * headDim};

                for (int tt{0}; tt < cacheLen[toSize(li)]; ++tt) {
                    auto dot{ops.leaf(0.0)};
                    if (!dot) {
                        return std::unexpected(dot.error());
                    }

                    for (int j{0}; j < headDim; ++j) {
                        const Ref qj{sc.q[toSize(hs + j)]};
                        const Ref kj{keyCache[toSize(li)][toSize(tt)][toSize(hs + j)]};

                        auto prod{ops.mul(qj, kj)};
                        if (!prod) {
                            return std::unexpected(prod.error());
                        }

                        auto sum{ops.add(*dot, *prod)};
                        if (!sum) {
                            return std::unexpected(sum.error());
                        }

                        dot = sum;
                    }

                    auto scale{ops.leaf(1.0 / std::sqrt(double(headDim)))};
                    if (!scale) {
                        return std::unexpected(scale.error());
                    }

                    auto scaled{ops.mul(*dot, *scale)};
                    if (!scaled) {
                        return std::unexpected(scaled.error());
                    }

                    sc.attnLogits[toSize(tt)] = *scaled;
                }

                double maxVal{-std::numeric_limits<double>::infinity()};
                for (int tt{0}; tt < cacheLen[toSize(li)]; ++tt) {
                    const double v0{ops.arena.getData(sc.attnLogits[toSize(tt)], ops.params)};
                    maxVal = std::max(maxVal, v0);
                }

                auto maxConst{ops.leaf(maxVal)};
                if (!maxConst) {
                    return std::unexpected(maxConst.error());
                }

                auto sumExp{ops.leaf(0.0)};
                if (!sumExp) {
                    return std::unexpected(sumExp.error());
                }

                for (int tt{0}; tt < cacheLen[toSize(li)]; ++tt) {
                    auto shifted{ops.sub(sc.attnLogits[toSize(tt)], *maxConst)};
                    if (!shifted) {
                        return std::unexpected(shifted.error());
                    }

                    auto e{ops.exp(*shifted)};
                    if (!e) {
                        return std::unexpected(e.error());
                    }

                    sc.attnWeights[toSize(tt)] = *e;

                    auto sum{ops.add(*sumExp, *e)};
                    if (!sum) {
                        return std::unexpected(sum.error());
                    }
                    sumExp = sum;
                }

                for (int tt{0}; tt < cacheLen[toSize(li)]; ++tt) {
                    auto divv{ops.div(sc.attnWeights[toSize(tt)], *sumExp)};
                    if (!divv) {
                        return std::unexpected(divv.error());
                    }
                    sc.attnWeights[toSize(tt)] = *divv;
                }

                for (int j{0}; j < headDim; ++j) {
                    auto acc{ops.leaf(0.0)};
                    if (!acc) {
                        return std::unexpected(acc.error());
                    }

                    for (int tt{0}; tt < cacheLen[toSize(li)]; ++tt) {
                        const Ref w{sc.attnWeights[toSize(tt)]};
                        const Ref vv{valueCache[toSize(li)][toSize(tt)][toSize(hs + j)]};

                        auto prod{ops.mul(w, vv)};
                        if (!prod) {
                            return std::unexpected(prod.error());
                        }

                        auto sum{ops.add(*acc, *prod)};
                        if (!sum) {
                            return std::unexpected(sum.error());
                        }

                        acc = sum;
                    }

                    sc.xAttn[toSize(hs + j)] = *acc;
                }
            }

            std::array<Ref, nEmbd> attnOut{};
            auto lo{linear(ops, off.attnWo[li], sc.xAttn, attnOut)};
            if (!lo) {
                return std::unexpected(lo.error());
            }

            for (int i{0}; i < nEmbd; ++i) {
                auto sum{ops.add(attnOut[toSize(i)], xResidual[toSize(i)])};
                if (!sum) {
                    return std::unexpected(sum.error());
                }
                sc.x[toSize(i)] = *sum;
            }

            const auto xResidual2{sc.x};

            auto rn2{rmsnorm(ops, sc.x, sc.tmp)};
            if (!rn2) {
                return std::unexpected(rn2.error());
            }
            sc.x = sc.tmp;

            for (int o{0}; o < 4 * nEmbd; ++o) {
                auto acc{ops.leaf(0.0)};
                if (!acc) {
                    return std::unexpected(acc.error());
                }

                const uint32_t rowBase{off.mlpFc1[li] + toU32(o * nEmbd)};
                for (int i{0}; i < nEmbd; ++i) {
                    const Ref w{makeParamRef(rowBase + toU32(i))};

                    auto prod{ops.mul(w, sc.x[toSize(i)])};
                    if (!prod) {
                        return std::unexpected(prod.error());
                    }

                    auto sum{ops.add(*acc, *prod)};
                    if (!sum) {
                        return std::unexpected(sum.error());
                    }

                    acc = sum;
                }

                sc.fc1[toSize(o)] = *acc;
            }

            for (int i{0}; i < 4 * nEmbd; ++i) {
                auto r{ops.relu(sc.fc1[toSize(i)])};
                if (!r) {
                    return std::unexpected(r.error());
                }
                sc.fc1[toSize(i)] = *r;
            }

            std::array<Ref, nEmbd> mlpOut{};
            for (int o{0}; o < nEmbd; ++o) {
                auto acc{ops.leaf(0.0)};
                if (!acc) {
                    return std::unexpected(acc.error());
                }

                const uint32_t rowBase{off.mlpFc2[li] + toU32(o * (4 * nEmbd))};
                for (int i{0}; i < 4 * nEmbd; ++i) {
                    const Ref w{makeParamRef(rowBase + toU32(i))};

                    auto prod{ops.mul(w, sc.fc1[toSize(i)])};
                    if (!prod) {
                        return std::unexpected(prod.error());
                    }

                    auto sum{ops.add(*acc, *prod)};
                    if (!sum) {
                        return std::unexpected(sum.error());
                    }

                    acc = sum;
                }

                mlpOut[toSize(o)] = *acc;
            }

            for (int i{0}; i < nEmbd; ++i) {
                auto sum{ops.add(mlpOut[toSize(i)], xResidual2[toSize(i)])};
                if (!sum) {
                    return std::unexpected(sum.error());
                }
                sc.x[toSize(i)] = *sum;
            }
        }

        auto ll{linearVocab(ops, off.lmHead, sc.x, sc.logits)};
        if (!ll) {
            return std::unexpected(ll.error());
        }

        return {};
    }
};

// Fast sampling (kept exactly as before)
int sampleCategoricalFast(const std::vector<double>& probs, std::mt19937& rng) noexcept {
    double sum{0.0};
    for (double p : probs) {
        sum += p;
    }

    if (!(sum > 0.0)) {
        std::uniform_int_distribution<int> uid{0, static_cast<int>(probs.size()) - 1};
        return uid(rng);
    }

    std::uniform_real_distribution<double> urd{0.0, sum};
    const double r{urd(rng)};

    double cdf{0.0};
    for (int i{0}; i < static_cast<int>(probs.size()); ++i) {
        cdf += probs[toSize(i)];
        if (r <= cdf) {
            return i;
        }
    }

    return static_cast<int>(probs.size()) - 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto tProgramStart{Clock::now()};

        RunMode runMode{RunMode::fast};
        std::string_view datasetPath{"input.txt"};

        for (int i{1}; i < argc; ++i) {
            const std::string_view arg{(argv[i] != nullptr) ? std::string_view{argv[i]} : std::string_view{}};

            if (arg == "--karpathy-py-compat") {
                runMode = RunMode::pythonCompat;
            } else if (arg == "--fast") {
                runMode = RunMode::fast;
            } else if (!arg.empty() && arg[0] != '-') {
                datasetPath = arg;
            }
        }

        std::mt19937 fastRng{42u};
        PythonRandom pyRng{42u};

        // ----------------------------- init (includes allocations) -----------------------------
        const auto tInitStart{Clock::now()};

        auto docsExp{readDocs(datasetPath)};
        if (!docsExp) {
            std::cerr << "error: " << docsExp.error().message << " (" << datasetPath << ")\n";
            return 1;
        }

        auto& docs{*docsExp};

        if (runMode == RunMode::pythonCompat) {
            pyRng.shuffle(docs);
        } else {
            std::shuffle(docs.begin(), docs.end(), fastRng);
        }

        std::cout << "num docs: " << docs.size() << "\n";

        auto tokExp{Tokenizer::build(docs)};
        if (!tokExp) {
            std::cerr << "error: " << tokExp.error().message << "\n";
            return 1;
        }

        const Tokenizer tokenizer{*tokExp};
        std::cout << "vocab size: " << tokenizer.vocabSize << "\n";

        Model model{};
        auto mi{model.init(tokenizer.vocabSize, tokenizer.bosId)};
        if (!mi) {
            std::cerr << "error: model init failed\n";
            return 1;
        }

        if (runMode == RunMode::pythonCompat) {
            model.initParamsPython(pyRng);
        } else {
            model.initParamsFast(fastRng);
        }

        std::cout << "num params: " << model.params.size() << "\n";

        Scratch sc{};
        auto si{sc.init(tokenizer.vocabSize)};
        if (!si) {
            std::cerr << "error: scratch init failed\n";
            return 1;
        }

        const uint32_t worstNodes{ArenaNodes::trainingStepWorst(static_cast<uint32_t>(tokenizer.vocabSize))};

        AutogradArena arena{};
        auto ai{arena.init(static_cast<size_t>(worstNodes), runMode == RunMode::pythonCompat)};
        if (!ai) {
            std::cerr << "error: arena init failed\n";
            return 1;
        }

        std::cout << "arena nodes (worst-case exact): " << worstNodes << "\n";

        using KvCache = Model::KvCache;
        KvCache keyCache{};
        KvCache valueCache{};
        std::array<int, nLayer> cacheLen{};
        std::array<int, blockSize + 2> tokens{};

        const auto tInitEnd{Clock::now()};

        // ----------------------------- training -----------------------------
        const auto tTrainStart{Clock::now()};

        double lossFirst{0.0};
        double lossLast{0.0};
        double lossMin{std::numeric_limits<double>::infinity()};
        double lossSum{0.0};
        uint32_t peakArenaUsed{0};

        for (int step{0}; step < numSteps; ++step) {
            arena.reset();
            model.zeroParamGrads();

            for (int li{0}; li < nLayer; ++li) {
                cacheLen[toSize(li)] = 0;
            }

            const std::string& doc{docs[toSize(step % static_cast<int>(docs.size()))]};

            tokens[0] = tokenizer.bosId;
            int len{1};

            for (char c : doc) {
                if (len >= blockSize + 1) {
                    break;
                }

                auto idExp{tokenizer.encodeChar(c)};
                if (!idExp) {
                    std::cerr << "error: " << idExp.error().message << "\n";
                    return 1;
                }

                tokens[toSize(len)] = *idExp;
                len += 1;
            }

            tokens[toSize(len)] = tokenizer.bosId;
            len += 1;

            const int n{std::min(blockSize, len - 1)};

            Ops ops{arena, model.params};

            auto lossSumRef{ops.leaf(0.0)};
            if (!lossSumRef) {
                std::cerr << "error: " << lossSumRef.error().message << "\n";
                return 1;
            }

            auto invN{ops.leaf(1.0 / double(n))};
            if (!invN) {
                std::cerr << "error: " << invN.error().message << "\n";
                return 1;
            }

            for (int posId{0}; posId < n; ++posId) {
                const int tokenId{tokens[toSize(posId)]};
                const int targetId{tokens[toSize(posId + 1)]};

                auto fw{model.gptStep(ops, tokenId, posId, keyCache, valueCache, cacheLen, sc)};
                if (!fw) {
                    std::cerr << "error: " << fw.error().message << "\n";
                    return 1;
                }

                auto sm{model.softmax(ops, sc.logits, sc.probs)};
                if (!sm) {
                    std::cerr << "error: " << sm.error().message << "\n";
                    return 1;
                }

                const Ref pt{sc.probs[toSize(targetId)]};

                auto logPt{ops.log(pt)};
                if (!logPt) {
                    std::cerr << "error: " << logPt.error().message << "\n";
                    return 1;
                }

                auto negLogPt{ops.neg(*logPt)};
                if (!negLogPt) {
                    std::cerr << "error: " << negLogPt.error().message << "\n";
                    return 1;
                }

                auto sum{ops.add(*lossSumRef, *negLogPt)};
                if (!sum) {
                    std::cerr << "error: " << sum.error().message << "\n";
                    return 1;
                }

                lossSumRef = sum;
            }

            auto lossMean{ops.mul(*lossSumRef, *invN)};
            if (!lossMean) {
                std::cerr << "error: " << lossMean.error().message << "\n";
                return 1;
            }

            const uint32_t lossNode{refIndex(*lossMean)};

            if (runMode == RunMode::pythonCompat) {
                arena.backwardFromPythonTopo(lossNode, model.paramGrads);
            } else {
                arena.backwardFromCreationOrder(lossNode, model.paramGrads);
            }

            model.adamStep(step);

            const double lossVal{arena.node(lossNode).data};
            if (step == 0) {
                lossFirst = lossVal;
            }
            lossLast = lossVal;
            lossMin = std::min(lossMin, lossVal);
            lossSum += lossVal;

            peakArenaUsed = std::max(peakArenaUsed, arena.size());

            std::cout << "step " << std::setw(4) << (step + 1) << " / " << std::setw(4) << numSteps
                      << " | loss " << std::fixed << std::setprecision(4) << lossVal << "\n";
        }

        const auto tTrainEnd{Clock::now()};

        // ----------------------------- inference -----------------------------
        const auto tInferStart{Clock::now()};

        std::cout << "\n--- inference (new, hallucinated names) ---\n";

        uint64_t totalGeneratedChars{0u};

        for (int sampleIdx{0}; sampleIdx < 20; ++sampleIdx) {
            arena.reset();

            for (int li{0}; li < nLayer; ++li) {
                cacheLen[toSize(li)] = 0;
            }

            Ops ops{arena, model.params};

            int tokenId{tokenizer.bosId};
            std::array<char, blockSize + 1> out{};
            int outLen{0};

            for (int posId{0}; posId < blockSize; ++posId) {
                auto fw{model.gptStep(ops, tokenId, posId, keyCache, valueCache, cacheLen, sc)};
                if (!fw) {
                    std::cerr << "error: " << fw.error().message << "\n";
                    return 1;
                }

                double maxVal{-std::numeric_limits<double>::infinity()};
                for (int i{0}; i < tokenizer.vocabSize; ++i) {
                    const double v0{arena.getData(sc.logits[toSize(i)], model.params) / temperature};
                    maxVal = std::max(maxVal, v0);
                }

                double sumExp{0.0};
                for (int i{0}; i < tokenizer.vocabSize; ++i) {
                    const double v0{arena.getData(sc.logits[toSize(i)], model.params) / temperature};
                    const double e{std::exp(v0 - maxVal)};
                    sc.probsData[toSize(i)] = e;
                    sumExp += e;
                }

                const double invSum{1.0 / sumExp};
                for (double& w : sc.probsData) {
                    w *= invSum;
                }

                if (runMode == RunMode::pythonCompat) {
                    tokenId = static_cast<int>(pyRng.choicesIndex(sc.probsData));
                } else {
                    tokenId = sampleCategoricalFast(sc.probsData, fastRng);
                }

                if (tokenId == tokenizer.bosId) {
                    break;
                }

                out[toSize(outLen)] = tokenizer.decodeId(tokenId);
                outLen += 1;
            }

            out[toSize(outLen)] = '\0';
            totalGeneratedChars += static_cast<uint64_t>(outLen);

            std::cout << "sample " << std::setw(2) << (sampleIdx + 1) << ": " << out.data() << "\n";
        }

        const auto tInferEnd{Clock::now()};
        const auto tProgramEnd{Clock::now()};

        // ----------------------------- summary stats (aligned, no heap) -----------------------------
        const auto initDur{tInitEnd - tInitStart};
        const auto trainDur{tTrainEnd - tTrainStart};
        const auto inferDur{tInferEnd - tInferStart};
        const auto totalDur{tProgramEnd - tProgramStart};

        const double lossAvg{lossSum / double(numSteps)};

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
        std::cout << ((runMode == RunMode::pythonCompat) ? "python-compat" : "fast") << "\n";

        printKey("dataset path");
        std::cout << datasetPath << "\n";

        printMs("init (includes allocations)", initDur);
        printMs("training", trainDur);
        printMs("inference", inferDur);
        printMs("total (init+train+infer)", totalDur);
        printMs("train+infer (no init)", trainDur + inferDur);

        printUs("training per step", toMicroseconds(trainDur) / double(numSteps));
        printUs("inference per sample", toMicroseconds(inferDur) / 20.0);

        printKey("loss");
        std::cout << std::fixed << std::setprecision(4)
                  << "first=" << lossFirst
                  << " last=" << lossLast
                  << " min=" << lossMin
                  << " avg=" << lossAvg << "\n";

        printKey("arena peak used (nodes)");
        std::cout << std::right << std::setw(12) << peakArenaUsed << " / " << worstNodes << "\n";

        printKey("generated chars (total)");
        std::cout << std::right << std::setw(12) << totalGeneratedChars << "\n";

        return 0;
    } catch (const std::bad_alloc&) {
        std::cerr << "fatal: out of memory during initialization\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "fatal init exception: " << e.what() << "\n";
        return 1;
    }
}
