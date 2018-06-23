/*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

#ifndef _HPP_BB_KV_CONTEXT
#define _HPP_BB_KV_CONTEXT

#include "bb_context.h"
#include "../bb_util.hpp"

namespace bb {
  namespace kv {

    template<typename K, typename T>
    SortContext<K, T>::SortContext(int n, int segments_length) {
      init(n, segments_length);
    }

    template<typename K, typename T>
    SortContext<K, T>::~SortContext() {
      free();
    }

    template<typename K, typename T>
    int SortContext<K, T>::init(int n, int segments_length) {
      cudaError_t cuda_err;
      h_bin_counter = new int[SEGBIN_NUM];

      cuda_err = cudaMalloc((void **) &d_bin_counter, SEGBIN_NUM * sizeof(int));
      CUDA_CHECK(cuda_err, "alloc d_bin_counter");
      cuda_err = cudaMalloc((void **) &d_bin_segs_id, segments_length * sizeof(int));
      CUDA_CHECK(cuda_err, "alloc d_bin_segs_id");

      cuda_err = cudaMalloc((void **) &keysB_d, n * sizeof(K));
      CUDA_CHECK(cuda_err, "alloc keysB_d");
      cuda_err = cudaMalloc((void **) &valsB_d, n * sizeof(T));
      CUDA_CHECK(cuda_err, "alloc valsB_d");

      return 0;
    }

    template<typename K, typename T>
    int SortContext<K, T>::free() {
      cudaError_t cuda_err;

      cuda_err = cudaFree(d_bin_counter);
      CUDA_CHECK(cuda_err, "free d_bin_counter");
      cuda_err = cudaFree(d_bin_segs_id);
      CUDA_CHECK(cuda_err, "free d_bin_segs_id");
      cuda_err = cudaFree(keysB_d);
      CUDA_CHECK(cuda_err, "free keysB");
      cuda_err = cudaFree(valsB_d);
      CUDA_CHECK(cuda_err, "free valsB");

      delete[] h_bin_counter;
      return 0;
    }

  }
}

#endif
