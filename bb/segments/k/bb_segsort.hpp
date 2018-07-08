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

#ifndef _HPP_BB_SEGMENTS_K_SEGSORT
#define _HPP_BB_SEGMENTS_K_SEGSORT

#include "../../k/bb_context.hpp"
#include "../bb_bin.hpp"
#include "bb_segsort.h"
#include "bb_comput_s.hpp"
#include "bb_comput_l.hpp"

namespace bb {
  namespace segments {
    namespace k {
      using namespace bb::k;

      template<class K>
      int bb_segsort(K *keys_d, int n, int *d_segs, int length, const SortContext<K> *context) {
        cudaError_t cuda_err;
        int *h_bin_counter = context->h_bin_counter;

        int *d_bin_counter = context->d_bin_counter;
        int *d_bin_segs_id = context->d_bin_segs_id;

        cuda_err = cudaMemset(d_bin_counter, 0, SEGBIN_NUM * sizeof(int));
        CUDA_CHECK(cuda_err, "memset d_bin_counter");

        K *keysB_d = context->keysB_d;

        bb_bin(d_bin_segs_id, d_bin_counter, d_segs, length, n, h_bin_counter);

        cudaStream_t streams[SEGBIN_NUM - 1];
        for (int i = 0; i < SEGBIN_NUM - 1; i++) cudaStreamCreate(&streams[i]);

        int subwarp_size, subwarp_num, factor;
        dim3 blocks(256, 1, 1);
        dim3 grids(1, 1, 1);

        blocks.x = 256;
        subwarp_num = h_bin_counter[1] - h_bin_counter[0];
        grids.x = (subwarp_num + blocks.x - 1) / blocks.x;
        if (subwarp_num > 0)
          gen_copy << < grids, blocks, 0, streams[0] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[0], subwarp_num, length);

        blocks.x = 256;
        subwarp_size = 2;
        subwarp_num = h_bin_counter[2] - h_bin_counter[1];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk256_wp2_tc1_r2_r2_orig << < grids, blocks, 0, streams[1] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[1], subwarp_num, length);

        blocks.x = 128;
        subwarp_size = 2;
        subwarp_num = h_bin_counter[3] - h_bin_counter[2];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk128_wp2_tc2_r3_r4_orig << < grids, blocks, 0, streams[2] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[2], subwarp_num, length);

        blocks.x = 128;
        subwarp_size = 2;
        subwarp_num = h_bin_counter[4] - h_bin_counter[3];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk128_wp2_tc4_r5_r8_orig << < grids, blocks, 0, streams[3] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[3], subwarp_num, length);

        blocks.x = 128;
        subwarp_size = 4;
        subwarp_num = h_bin_counter[5] - h_bin_counter[4];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk128_wp4_tc4_r9_r16_strd << < grids, blocks, 0, streams[4] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[4], subwarp_num, length);

        blocks.x = 128;
        subwarp_size = 8;
        subwarp_num = h_bin_counter[6] - h_bin_counter[5];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk128_wp8_tc4_r17_r32_strd << < grids, blocks, 0, streams[5] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[5], subwarp_num, length);

        blocks.x = 128;
        subwarp_size = 16;
        subwarp_num = h_bin_counter[7] - h_bin_counter[6];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk128_wp16_tc4_r33_r64_strd << < grids, blocks, 0, streams[6] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[6], subwarp_num, length);

        blocks.x = 256;
        subwarp_size = 8;
        subwarp_num = h_bin_counter[8] - h_bin_counter[7];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk256_wp8_tc16_r65_r128_strd << < grids, blocks, 0, streams[7] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[7], subwarp_num, length);

        blocks.x = 256;
        subwarp_size = 32;
        subwarp_num = h_bin_counter[9] - h_bin_counter[8];
        factor = blocks.x / subwarp_size;
        grids.x = (subwarp_num + factor - 1) / factor;
        if (subwarp_num > 0)
          gen_bk256_wp32_tc8_r129_r256_strd << < grids, blocks, 0, streams[8] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[8], subwarp_num, length);

        blocks.x = 128;
        subwarp_num = h_bin_counter[10] - h_bin_counter[9];
        grids.x = subwarp_num;
        if (subwarp_num > 0)
          gen_bk128_tc4_r257_r512_orig << < grids, blocks, 0, streams[9] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[9], subwarp_num, length);

        blocks.x = 256;
        subwarp_num = h_bin_counter[11] - h_bin_counter[10];
        grids.x = subwarp_num;
        if (subwarp_num > 0)
          gen_bk256_tc4_r513_r1024_orig << < grids, blocks, 0, streams[10] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[10], subwarp_num, length);

        blocks.x = 512;
        subwarp_num = h_bin_counter[12] - h_bin_counter[11];
        grids.x = subwarp_num;
        if (subwarp_num > 0)
          gen_bk512_tc4_r1025_r2048_orig << < grids, blocks, 0, streams[11] >> > (keys_d, keysB_d,
              n, d_segs, d_bin_segs_id + h_bin_counter[11], subwarp_num, length);

        // sort long segments
        subwarp_num = length - h_bin_counter[12];
        if (subwarp_num > 0)
          gen_grid_kern_r2049(keys_d, keysB_d,
                              n, d_segs, d_bin_segs_id + h_bin_counter[12], subwarp_num, length);

        // std::swap(keys_d, keysB_d);
        cuda_err = cudaMemcpy(keys_d, keysB_d, sizeof(K) * n, cudaMemcpyDeviceToDevice);
        CUDA_CHECK(cuda_err, "copy to keys_d from keysB_d");

        for (int i = 0; i < SEGBIN_NUM - 1; i++) cudaStreamDestroy(streams[i]);

        return 1;
      }

      template<class K>
      int bb_segsort(K *keys_d, int n, int *d_segs, int length) {
        SortContext<K> context(n, length);
        return bb_segsort(keys_d, n, d_segs, length, &context);
      }

    }
  }
}
#endif
