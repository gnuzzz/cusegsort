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

#ifndef _H_BB_MATRIX_K_SEGSORT
#define _H_BB_MATRIX_K_SEGSORT

#include "../../k/bb_context.h"

namespace bb {
  namespace matrix {
    namespace k {
      using namespace bb::k;

      template<class K>
      int bb_segsort(K *keys_d, const int rows, const int cols, const SortContext<K> *context);

      template<class K>
      int bb_segsort(K *keys_d, const int rows, const int cols);

    }
  }
}

#endif
