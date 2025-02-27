#pragma once

#include <torch/extension.h>

#include <limits>
#include <map>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

__device__ __inline__ at::Half
__shfl_sync(const unsigned mask, const at::Half var, const int srcLane) {
  return __shfl_sync(mask, var, srcLane);
  // return __shfl_sync(mask, (__half)var, srcLane);
}

__device__ __inline__ at::Half __shfl_down_sync(const unsigned mask,
                                                const at::Half var,
                                                const unsigned int delta) {
  return __shfl_down_sync(mask, var, delta);
  // return __shfl_down_sync(mask, (__half)var, delta);
}

enum ReductionType { SUM, MEAN, MUL, DIV, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mean", MEAN}, {"mul", MUL},
    {"div", DIV}, {"min", MIN},   {"max", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)                               \
  [&] {                                                                        \
    switch (reduce2REDUCE.at(reduce)) {                                        \
    case SUM: {                                                                \
      const ReductionType REDUCE = SUM;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MEAN: {                                                               \
      const ReductionType REDUCE = MEAN;                                       \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MUL: {                                                                \
      const ReductionType REDUCE = MUL;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case DIV: {                                                                \
      const ReductionType REDUCE = DIV;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MIN: {                                                                \
      const ReductionType REDUCE = MIN;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case MAX: {                                                                \
      const ReductionType REDUCE = MAX;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    }                                                                          \
  }()

template <typename scalar_t, ReductionType REDUCE> struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline __host__ __device__ void update(scalar_t *val, scalar_t new_val,
                                                int64_t *arg, int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN && new_val < *val) ||
             (REDUCE == MAX && new_val > *val)) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline __host__ __device__ void write(scalar_t *address, scalar_t val,
                                               int64_t *arg_address,
                                               int64_t arg, int count) {
    if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
      *address = val;
    else if (REDUCE == MEAN)
      *address = val / (scalar_t)(count > 0 ? count : 1);
    else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else
        *address = (scalar_t)0;
    }
  }
};