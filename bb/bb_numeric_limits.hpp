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

#include <float.h>

#ifndef _HPP_BB_NUMERIC_LIMITS
#define _HPP_BB_NUMERIC_LIMITS

namespace bb {

  namespace numeric_limits {

    template<class T>
    __host__ __device__
    T max();

    template<class T>
    __host__ __device__
    T min();

    template<>
    __host__ __device__
    unsigned char max<unsigned char>() {
      return ~(unsigned char)0;
    }

    template<>
    __host__ __device__
    unsigned char min<unsigned char>() {
      return (unsigned char)0;
    }

    template<>
    __host__ __device__
    char max<char>() {
      return (char) (max<unsigned char>() >> 1);
    }

    template<>
    __host__ __device__
    char min<char>() {
      return (char) (~max<char>());
    }

    template<>
    __host__ __device__
    unsigned short max<unsigned short>() {
      return ~(unsigned short)0;
    }

    template<>
    __host__ __device__
    unsigned short min<unsigned short>() {
      return (unsigned short)0;
    }

    template<>
    __host__ __device__
    short max<short>() {
      return (short) (max<unsigned short>() >> 1);
    }

    template<>
    __host__ __device__
    short min<short>() {
      return (short) (~max<short>());
    }

    template<>
    __host__ __device__
    unsigned int max<unsigned int>() {
      return ~(unsigned int)0;
    }

    template<>
    __host__ __device__
    unsigned int min<unsigned int>() {
      return (unsigned int)0;
    }

    template<>
    __host__ __device__
    int max<int>() {
      return (int) (max<unsigned int>() >> 1);
    }

    template<>
    __host__ __device__
    int min<int>() {
      return (int) (~max<int>());
    }

    template<>
    __host__ __device__
    unsigned long long int max<unsigned long long int>() {
      return ~(unsigned long long int)0;
    }

    template<>
    __host__ __device__
    unsigned long long int min<unsigned long long int>() {
      return (unsigned long long int)0;
    }

    template<>
    __host__ __device__
    long long int max<long long int>() {
      return (long long int) (max<unsigned long long int>() >> 1);
    }

    template<>
    __host__ __device__
    long long int min<long long int>() {
      return (long long int) (~max<long long int>());
    }

    template<>
    __host__ __device__
    float max<float>() {
      return FLT_MAX;
    }

    template<>
    __host__ __device__
    float min<float>() {
      return -FLT_MAX;
    }

    template<>
    __host__ __device__
    double max<double>() {
      return DBL_MAX;
    }

    template<>
    __host__ __device__
    double min<double>() {
      return -DBL_MAX;
    }

  }

}

#endif

