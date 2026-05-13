#pragma once

#include "base.h"

// Pure normalized Hadamard-32 rotation R = H32 / sqrt(32), applied once per
// quantization block.
//
// Implementation strategy (verbatim from haoyangli0109/vllm @ 8ff85fe, the
// last known-good version):
//   * H32 = H4 within thread (4 lanes) ⊗ H8 across the 8-thread group
//     (3 stages of __shfl_xor with steps {1,2,4}).
//   * fp32 scalar arithmetic on low/high halves of each fp16x2 lane (two
//     parity passes), so the compiler can schedule adds/subs and shuffles
//     freely.
//   * H is symmetric and self-inverse for normalized Hadamard, so
//     forward == inverse.
//
// Layout assumption:
//   * One quantization block = 8 threads (kThreadGroupSize) x 4 fp16x2 lanes.
//   * The low and high halves of each fp16x2 form two independent 32-d blocks.

namespace quickreduce {

template <typename T>
__quickreduce_device_inline__ T float_to_T(float v);

template <>
__quickreduce_device_inline__ half float_to_T<half>(float v) {
  return __float2half_rn(v);
}

template <>
__quickreduce_device_inline__ nv_bfloat16 float_to_T<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__quickreduce_device_inline__ void rotate_group32_hadamard(int32x4_t& atom) {
  constexpr float kInvSqrt32 = 0.1767766952966369f;  // 1/sqrt(32)
  T* vals = reinterpret_cast<T*>(&atom);

#pragma unroll
  for (int parity = 0; parity < 2; ++parity) {
    float v0 = T2float_cast(vals[parity + 0 * 2]);
    float v1 = T2float_cast(vals[parity + 1 * 2]);
    float v2 = T2float_cast(vals[parity + 2 * 2]);
    float v3 = T2float_cast(vals[parity + 3 * 2]);

    // H4 within thread (unnormalized).
    float t0 = v0 + v1;
    float t1 = v0 - v1;
    float t2 = v2 + v3;
    float t3 = v2 - v3;
    float w0 = t0 + t2;
    float w1 = t1 + t3;
    float w2 = t0 - t2;
    float w3 = t1 - t3;

    float vec[4] = {w0, w1, w2, w3};

    // H8 across thread-group via xor-shuffle butterfly (unnormalized).
    // For each step s the pair (l_low, l_high = l_low | s) computes:
    //   x_new[l_low]  = x[l_low] + x[l_high]
    //   x_new[l_high] = x[l_low] - x[l_high]   (== partner - x == y - x)
    // The high lane MUST do y - x, not x - y, otherwise the implementation
    // computes D8*H8 (where D8 = diag((-1)^popcount(l))). Single application
    // is still orthogonal, but (D8*H8)^2 != I -- the round trip becomes a
    // sign-flipped reverse permutation across the 8-thread group, which
    // destroys accuracy for INT3+TurboQuant.
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float x = vec[c];
#pragma unroll
      for (int step = 1; step < kThreadGroupSize; step <<= 1) {
        float y = __shfl_xor(x, step, kThreadGroupSize);
        x = (threadIdx.x & step) ? (y - x) : (x + y);
      }
      vec[c] = x * kInvSqrt32;
    }

    vals[parity + 0 * 2] = float_to_T<T>(vec[0]);
    vals[parity + 1 * 2] = float_to_T<T>(vec[1]);
    vals[parity + 2 * 2] = float_to_T<T>(vec[2]);
    vals[parity + 3 * 2] = float_to_T<T>(vec[3]);
  }
}

// H is symmetric, normalized, and (with the y - x fix above) self-inverse.
template <typename T>
__quickreduce_device_inline__ void rotate_forward(int32x4_t& atom, int /*thread*/) {
  rotate_group32_hadamard<T>(atom);
}

template <typename T>
__quickreduce_device_inline__ void rotate_inverse(int32x4_t& atom, int /*thread*/) {
  rotate_group32_hadamard<T>(atom);
}

}  // namespace quickreduce
