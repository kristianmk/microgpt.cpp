cmake_minimum_required(VERSION 3.20)

project(GptAtomic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(GptAtomic       microgpt.cpp)
add_executable(GptAtomicTurbo  microgpt_turbo.cpp)

# --- Release-ish config expression (covers single + multi-config generators) ---
set(_REL_CFG "$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:1>")

# --- Optimization flags (portable-ish, non-MSVC) ---
if (NOT MSVC)
  target_compile_options(GptAtomicTurbo PRIVATE
    $<${_REL_CFG}:-O3 -DNDEBUG>
  )
endif()

# --- CPU tuning: probe flags instead of guessing compiler/target ---
include(CheckCXXCompilerFlag)

if (NOT MSVC)
  check_cxx_compiler_flag("-march=native" _HAS_MARCH_NATIVE)
  check_cxx_compiler_flag("-mcpu=native"  _HAS_MCPU_NATIVE)

  if (_HAS_MARCH_NATIVE)
    target_compile_options(GptAtomicTurbo PRIVATE $<${_REL_CFG}:-march=native>)
  elseif (_HAS_MCPU_NATIVE)
    target_compile_options(GptAtomicTurbo PRIVATE $<${_REL_CFG}:-mcpu=native>)
  endif()
endif()

# --- LTO/IPO (guarded) ---
include(CheckIPOSupported)
check_ipo_supported(RESULT _IPO_OK OUTPUT _IPO_ERR)
if (_IPO_OK)
  # Prefer config-specific property when available.
  set_property(TARGET GptAtomicTurbo PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
  message(STATUS "IPO/LTO not supported: ${_IPO_ERR}")
endif()

# Optional: MAX speed (numerics may change!)
option(GPTATOMIC_TURBO_FASTMATH "Enable fast-math for GptAtomicTurbo" OFF)
if (GPTATOMIC_TURBO_FASTMATH AND NOT MSVC)
  check_cxx_compiler_flag("-ffast-math"        _HAS_FFAST_MATH)
  check_cxx_compiler_flag("-fno-math-errno"    _HAS_FNO_MATH_ERRNO)
  check_cxx_compiler_flag("-ffp-contract=fast" _HAS_FFP_CONTRACT_FAST)

  if (_HAS_FFAST_MATH)
    target_compile_options(GptAtomicTurbo PRIVATE $<${_REL_CFG}:-ffast-math>)
  endif()
  if (_HAS_FNO_MATH_ERRNO)
    target_compile_options(GptAtomicTurbo PRIVATE $<${_REL_CFG}:-fno-math-errno>)
  endif()
  if (_HAS_FFP_CONTRACT_FAST)
    target_compile_options(GptAtomicTurbo PRIVATE $<${_REL_CFG}:-ffp-contract=fast>)
  endif()
endif()

# Optional: loop unrolling (benchmark!)
option(GPTATOMIC_TURBO_UNROLL "Enable loop unrolling for GptAtomicTurbo" OFF)
if (GPTATOMIC_TURBO_UNROLL AND NOT MSVC)
  check_cxx_compiler_flag("-funroll-loops" _HAS_FUNROLL_LOOPS)
  if (_HAS_FUNROLL_LOOPS)
    target_compile_options(GptAtomicTurbo PRIVATE $<${_REL_CFG}:-funroll-loops>)
  endif()
endif()

# --- dataset: allow overriding path, optional download at configure-time, verify SHA256 ---

set(GPT_ATOMIC_DATASET_URL
  "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
  CACHE STRING "Dataset URL")

set(GPT_ATOMIC_DATASET_SHA256
  "0a30b5557f192f32ab962680889aac5f6fda0f4cecf40a6d0b5694f58ea8cc4d"
  CACHE STRING "Expected SHA256 for the dataset")

# Pass with: -DGPT_ATOMIC_DATASET_PATH=/path/to/names.txt
set(GPT_ATOMIC_DATASET_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/input.txt"
  CACHE FILEPATH "Dataset file path (existing file or download destination)")

option(GPT_ATOMIC_DOWNLOAD_DATASET
  "Download dataset at configure time if GPT_ATOMIC_DATASET_PATH doesn't exist" ON)

# Normalize to an absolute path
get_filename_component(_DATA_FILE "${GPT_ATOMIC_DATASET_PATH}" ABSOLUTE)

# Download if missing (optional), otherwise verify SHA256
if (NOT EXISTS "${_DATA_FILE}")
  if (GPT_ATOMIC_DOWNLOAD_DATASET)
    message(STATUS "Downloading dataset to: ${_DATA_FILE}")
    file(DOWNLOAD "${GPT_ATOMIC_DATASET_URL}" "${_DATA_FILE}"
      TLS_VERIFY ON
      EXPECTED_HASH "SHA256=${GPT_ATOMIC_DATASET_SHA256}"
      SHOW_PROGRESS
      STATUS dl_status
    )
    list(GET dl_status 0 dl_code)
    if (NOT dl_code EQUAL 0)
      list(GET dl_status 1 dl_msg)
      message(FATAL_ERROR "Dataset download failed: ${dl_code} (${dl_msg})")
    endif()
  else()
    message(FATAL_ERROR
      "Dataset missing: ${_DATA_FILE}\n"
      "Provide -DGPT_ATOMIC_DATASET_PATH=/path/to/file (and SHA/URL if different),\n"
      "or enable -DGPT_ATOMIC_DOWNLOAD_DATASET=ON."
    )
  endif()
endif()

file(SHA256 "${_DATA_FILE}" have_sha256)
string(TOLOWER "${have_sha256}" have_sha256)
string(TOLOWER "${GPT_ATOMIC_DATASET_SHA256}" expect_sha256)
if (NOT have_sha256 STREQUAL expect_sha256)
  message(FATAL_ERROR
    "Dataset SHA256 mismatch:\n"
    "  have     = ${have_sha256}\n"
    "  expected = ${expect_sha256}\n"
    "  file     = ${_DATA_FILE}"
  )
endif()

# Compile-time default dataset path (CLI arg can override at runtime)
# Escape backslashes for safety (mostly relevant on Windows, harmless elsewhere).
set(_DATA_FILE_ESC "${_DATA_FILE}")
string(REPLACE "\\" "\\\\" _DATA_FILE_ESC "${_DATA_FILE_ESC}")

target_compile_definitions(GptAtomic      PRIVATE GPT_ATOMIC_DATASET_PATH=\"${_DATA_FILE_ESC}\")
target_compile_definitions(GptAtomicTurbo PRIVATE GPT_ATOMIC_DATASET_PATH=\"${_DATA_FILE_ESC}\")

# --- install ---
include(GNUInstallDirs)
install(TARGETS GptAtomic      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS GptAtomicTurbo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
