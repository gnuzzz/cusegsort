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

#include <iostream>
#include <stdio.h>
#include <algorithm>

#include "bb/segments/kv/bb_segsort.hpp"
#include "bb/segments/k/bb_segsort.hpp"
#include "bb/matrix/kv/bb_segsort.hpp"
#include "bb/matrix/k/bb_segsort.hpp"
#include "timer.h"

using namespace bb;

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }

int sort_segments(int* key, double* val, int* seg, int n, int length, const bb::kv::SortContext<int, double>* context) {
  cudaError_t err;
  int    *key_d;
  double *val_d;
  int    *seg_d;

  StartTimer();
  err = cudaMalloc((void**)&key_d, sizeof(int   )*n);
  CUDA_CHECK(err, "segments: alloc key_d");
  err = cudaMalloc((void**)&val_d, sizeof(double)*n);
  CUDA_CHECK(err, "segments: alloc val_d");
  err = cudaMalloc((void**)&seg_d, sizeof(int   )*length);
  CUDA_CHECK(err, "segments: alloc seg_d");
  const double mallocTime = GetTimer() / 1000.0;
  printf("segments malloc time: %f s\n", mallocTime);

  StartTimer();
  err = cudaMemcpy(key_d, key, sizeof(int   )*n, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "segments: copy to key_d");
  err = cudaMemcpy(val_d, val, sizeof(double)*n, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "segments: copy to val_d");
  err = cudaMemcpy(seg_d, seg, sizeof(int)*length, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "segments: copy to seg_d");
  const double memcpyTime = GetTimer() / 1000.0;
  printf("segments copy time: %f s\n", memcpyTime);

  StartTimer();
  bb::segments::kv::bb_segsort(key_d, val_d, n, seg_d, length, context);
  const double sortTime = GetTimer() / 1000.0;
  printf("segments sort time: %f s\n", sortTime);

  err = cudaFree(key_d);
  CUDA_CHECK(err, "segments: free key_d");
  err = cudaFree(val_d);
  CUDA_CHECK(err, "segments: free val_d");
  err = cudaFree(seg_d);
  CUDA_CHECK(err, "segments: free seg_d");

  return 0;
}

int sort_segments_keys(int* key, int* seg, int n, int length, const bb::k::SortContext<int>* context) {
  cudaError_t err;
  int    *key_d;
  int    *seg_d;

  StartTimer();
  err = cudaMalloc((void**)&key_d, sizeof(int   )*n);
  CUDA_CHECK(err, "segments: alloc key_d");
  err = cudaMalloc((void**)&seg_d, sizeof(int   )*length);
  CUDA_CHECK(err, "segments: alloc seg_d");
  const double mallocTime = GetTimer() / 1000.0;
  printf("segments malloc time: %f s\n", mallocTime);

  StartTimer();
  err = cudaMemcpy(key_d, key, sizeof(int   )*n, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "segments: copy to key_d");
  err = cudaMemcpy(seg_d, seg, sizeof(int)*length, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "segments: copy to seg_d");
  const double memcpyTime = GetTimer() / 1000.0;
  printf("segments copy time: %f s\n", memcpyTime);

  StartTimer();
  bb::segments::k::bb_segsort(key_d, n, seg_d, length, context);
  const double sortTime = GetTimer() / 1000.0;
  printf("segments sort time: %f s\n", sortTime);

  err = cudaFree(key_d);
  CUDA_CHECK(err, "segments: free key_d");
  err = cudaFree(seg_d);
  CUDA_CHECK(err, "segments: free seg_d");

  return 0;
}

int sort_matrix(int* key, double* val, int rows, int cols, const bb::kv::SortContext<int, double>* context) {
  cudaError_t err;
  int    *key_d;
  double *val_d;
  int n = rows * cols;

  StartTimer();
  err = cudaMalloc((void**)&key_d, sizeof(int   )*n);
  CUDA_CHECK(err, "matrix: alloc key_d");
  err = cudaMalloc((void**)&val_d, sizeof(double)*n);
  CUDA_CHECK(err, "matrix: alloc val_d");
  const double mallocTime = GetTimer() / 1000.0;
  printf("matrix malloc time: %f s\n", mallocTime);

  StartTimer();
  err = cudaMemcpy(key_d, key, sizeof(int   )*n, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "matrix: copy to key_d");
  err = cudaMemcpy(val_d, val, sizeof(double)*n, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "matrix: copy to val_d");
  const double memcpyTime = GetTimer() / 1000.0;
  printf("matrix copy time: %f s\n", memcpyTime);

  //show_d(key_d, n, "keys before sort:\n");
  //show_d(val_d, n, "vals before sort:\n");
  StartTimer();
  bb::matrix::kv::bb_segsort(key_d, val_d, rows, cols, context);
  const double sortTime = GetTimer() / 1000.0;
  printf("matrix sort time: %f s\n", sortTime);
  //show_d(key_d, n, "keys after sort:\n");
  //show_d(val_d, n, "vals after sort:\n");

  err = cudaFree(key_d);
  CUDA_CHECK(err, "matrix: free key_d");
  err = cudaFree(val_d);
  CUDA_CHECK(err, "matrix: free val_d");

  return 0;
}

int sort_matrix_keys(int* key, int rows, int cols, const bb::k::SortContext<int>* context) {
  cudaError_t err;
  int    *key_d;
  int n = rows * cols;

  StartTimer();
  err = cudaMalloc((void**)&key_d, sizeof(int   )*n);
  CUDA_CHECK(err, "matrix: alloc key_d");
  const double mallocTime = GetTimer() / 1000.0;
  printf("matrix malloc time: %f s\n", mallocTime);

  StartTimer();
  err = cudaMemcpy(key_d, key, sizeof(int   )*n, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "matrix: copy to key_d");
  const double memcpyTime = GetTimer() / 1000.0;
  printf("matrix copy time: %f s\n", memcpyTime);

  //show_d(key_d, n, "keys before sort:\n");
  //show_d(val_d, n, "vals before sort:\n");
  StartTimer();
  bb::matrix::k::bb_segsort(key_d, rows, cols, context);
  const double sortTime = GetTimer() / 1000.0;
  printf("matrix sort time: %f s\n", sortTime);
  //show_d(key_d, n, "keys after sort:\n");
  //show_d(val_d, n, "vals after sort:\n");

  err = cudaFree(key_d);
  CUDA_CHECK(err, "matrix: free key_d");

  return 0;
}

int sort() {

  int rows = 8000;
//  int rows = 4;
  int cols = 5000;
//  int cols = 5;
  int n = rows * cols;
  int* key = new int[n];
  double* val = new double[n];
  int* seg = new int[rows];

  StartTimer();
  for (int i = 0; i < n; i++) {
    key[i] = (rand()%(n-1-0+1)+0);
  }
  for (int i = 0; i < n; i++) {
    val[i] = (double)(rand()%(n-1-0+1)+0);
  }
  for (int i = 0; i < rows; i++)
    seg[i] = i * cols;
  const double initDataTime = GetTimer() / 1000.0;
  printf("Init data time: %f s\n", initDataTime);

  bb::kv::SortContext<int, double> context_kv(rows * cols, rows);

//  StartTimer();
  for (int i = 0; i < 3; i++) {
    sort_segments(key, val, seg, n, rows, &context_kv);
  }
//  const double sortSegmentsTime = GetTimer() / 1000.0 / 3.0;

//  StartTimer();
  for (int i = 0; i < 3; i++) {
    sort_matrix(key, val, rows, cols, &context_kv);
  }
//  const double sortMatrixTime = GetTimer() / 1000.0 / 3.0;

  bb::k::SortContext<int> context_k(rows * cols, rows);

  for (int i = 0; i < 3; i++) {
    sort_segments_keys(key, seg, n, rows, &context_k);
  }

  for (int i = 0; i < 3; i++) {
    sort_matrix_keys(key, rows, cols, &context_k);
  }

//  printf("Avg sort segments time: %f s\n", sortSegmentsTime);
//  printf("Avg sort matrix time: %f s\n", sortMatrixTime);

  delete seg;
  delete val;
  delete key;

  return 0;
}

int main() {
  sort();
  return 0;
}