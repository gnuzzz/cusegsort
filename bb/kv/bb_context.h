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

#ifndef _H_BB_KV_CONTEXT
#define _H_BB_KV_CONTEXT

namespace bb {
  namespace kv {

    template<typename K, typename T>
    struct SortContext {

      int *h_bin_counter;
      int *d_bin_counter;
      int *d_bin_segs_id;
      K *keysB_d;
      T *valsB_d;

      SortContext(int n, int segments_length);
      ~SortContext();

    private:
      int init(int n, int segments_length);
      int free();

    };

  }
}

#endif
