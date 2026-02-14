# microgpt.cpp

Small, single-file(s) C++23 implementations of a tiny character-level GPT model based on Andrej Karpathy’s microgpt [https://karpathy.github.io/2026/02/12/microgpt/]. Mostly written by ChatGPT 5.2 thinking and Claude Opus 4.6 Extended. Not completely true to the 200-lines of Python code “art project”, but much faster and can optionally produce bitwise-identical output to the original.

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

## Example output from ./GptAtomicTurbo
```bash
--- inference (new, hallucinated names) ---
sample  1: vilya
sample  2: damiien
sample  3: sana
sample  4: maran
sample  5: sonelo
sample  6: nahela
sample  7: mayania
sample  8: atelin
sample  9: karina
sample 10: rielan
sample 11: jene
sample 12: donis
sample 13: harana
sample 14: anele
sample 15: lannie
sample 16: raey
sample 17: adyney
sample 18: jama
sample 19: jarien
sample 20: zaria

generated chars (total)                    109

--- run statistics ---
mode                              turbo-kernels
dataset path                      input.txt
steps                             1000
tokens per step (last)            7
init (includes allocations)               2.81 ms
training                                 21.93 ms
inference                                 0.19 ms
total (init+train+infer)                 24.93 ms
train+infer (no init)                    22.12 ms
training per step                        21.93 us
inference per sample                      9.52 us
loss                              first=3.6024 last=2.8826 min=1.5387 avg=2.4558
```


## Example output from ./GptAtomic --karpathy-py-compat
```bash
--- inference (new, hallucinated names) ---
sample  1: kamon
sample  2: ann
sample  3: karai
sample  4: jaire
sample  5: vialan
sample  6: karia
sample  7: yeran
sample  8: anna
sample  9: areli
sample 10: kaina
sample 11: konna
sample 12: keylen
sample 13: liole
sample 14: alerin
sample 15: earan
sample 16: lenne
sample 17: kana
sample 18: lara
sample 19: alela
sample 20: anton

--- run statistics ---
mode                              python-compat
dataset path                      input.txt
init (includes allocations)               3.92 ms
training                                598.94 ms
inference                                 2.09 ms
total (init+train+infer)                604.95 ms
train+infer (no init)                   601.03 ms
training per step                       598.94 us
inference per sample                    104.33 us
loss                              first=3.3660 last=2.6497 min=1.5782 avg=2.4517
arena peak used (nodes)                 116721 / 134931
generated chars (total)                     98
```
