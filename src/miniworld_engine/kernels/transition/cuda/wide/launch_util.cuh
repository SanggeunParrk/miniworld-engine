// Shared by the launcher translation units: the one-time opt-in to more than 48 KB of dynamic shared memory.
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void wide_smem_optin(const void* fn, int bytes, const char* what) {
  // Once per kernel per process. sm_90's effective dynamic ceiling is 231424 B, not the 232448 B the occupancy API advertises.
  cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + " smem opt-in: " + cudaGetErrorString(e));
}
