# microgpt.cpp

Small, single-file(s) C++23 implementations of a tiny character-level GPT model based on Andrej Karpathy’s microgpt [https://karpathy.github.io/2026/02/12/microgpt/].

Two executables are built:

- **GptAtomic**: reference implementation with an explicit autograd arena.
- **GptAtomicTurbo**: same model, but with higher performance CPU kernels and no autograd graph.

The default dataset is Karpathy’s `names.txt` from `makemore`. CMake downloads it at configure time and verifies a pinned SHA256.

## Requirements

- CMake 3.20+
- A C++23 compiler with `std::expected`
  - macOS: LLVM Clang from Homebrew is recommended
  - Ubuntu: GCC 13+ or Clang 17+

## Build

### macOS

```bash
brew install cmake ninja llvm

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++"

cmake --build build
```

### Ubuntu

```bash
sudo apt update
sudo apt install -y cmake ninja-build g++

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

Run from the build directory so `input.txt` is found:

```bash
cd build
./GptAtomic
./GptAtomicTurbo
```

You can also pass an explicit dataset path:

```bash
./GptAtomic /path/to/names.txt
./GptAtomicTurbo /path/to/names.txt
```

### Useful flags

**GptAtomic**

- `--fast` (default)
- `--karpathy-py-compat` for closer Python behavior (CPython RNG + Python topo order)

**GptAtomicTurbo**

- `--steps N` (default `1000`)
- `--samples N` (default `20`)
- `--log-every N` (default `25`)
- `--temperature T` (default `0.5`)

Example:

```bash
./GptAtomicTurbo --steps 2000 --samples 50 --temperature 0.8
```

## CMake options

- `-DGPTATOMIC_TURBO_FASTMATH=ON` enables fast-math flags for `GptAtomicTurbo`.
- `-DGPTATOMIC_TURBO_UNROLL=ON` enables loop unrolling for `GptAtomicTurbo`.
- `-DGPT_ATOMIC_DATASET_PATH=/path/to/file` overrides the dataset location.
- `-DGPT_ATOMIC_DOWNLOAD_DATASET=OFF` disables dataset download at configure time.

## License

MIT, see `LICENSE`.
