#pragma once
#include <c10/macros/Export.h>
#include <cstdint>
#include <c10/cuda/CUDAStream.h>

namespace at::cuda {

// enqueues a kernel that spins for the specified number of cycles
TORCH_CUDA_CU_API void sleep(int64_t cycles, at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream());

// flushes instruction cache for ROCm; no-op for CUDA
TORCH_CUDA_CU_API void flush_icache();

}  // namespace at::cuda
