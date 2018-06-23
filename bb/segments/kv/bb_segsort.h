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

#ifndef _H_BB_SEGMENTS_KV_SEGSORT
#define _H_BB_SEGMENTS_KV_SEGSORT

#include "../../kv/bb_context.h"

namespace bb {
  namespace segments {
    namespace kv {
      using namespace bb::kv;

      template<class K, class T>
      int bb_segsort(K *keys_d, T *vals_d, int n, int *d_segs, int length, const SortContext<K, T> *context);

      template<class K, class T>
      int bb_segsort(K *keys_d, T *vals_d, int n, int *d_segs, int length);
    }
  }
}

#endif
