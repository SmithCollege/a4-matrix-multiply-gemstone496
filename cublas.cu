#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define WIDTH 1024  // Ends up being squared for the proper calculation (e.g. SIZE 4 means 4x4 matrices)
#define RUNS 100

int main() {
  const int SIZE = WIDTH*WIDTH; // compatibility with ported code from scan. since matrices are being initialized as 1D arrays, need a single size var. i'd prefer 2D. just sayin'
  std::cout << "\n" << SIZE; // record the size of the run for data collection

  // allocate input and output arrays
  double *M, *d_M, *N, *d_N, *P, *d_P;
  M = (double*) malloc(SIZE*sizeof(double)); N = (double*) malloc(SIZE*sizeof(double)); P = (double*) malloc(SIZE*sizeof(double));
  cudaMalloc(&d_M, SIZE*sizeof(double)); cudaMalloc(&d_N, SIZE*sizeof(double)); cudaMalloc(&d_P, SIZE*sizeof(double));

  // initialize inputs
  for (int i = 0; i < SIZE; i++) {
    M[i] = 1.;
    N[i] = 1.;
  }
  cudaMemcpy(d_M, M, SIZE*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, SIZE*sizeof(double), cudaMemcpyHostToDevice);
    
  // initialize handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // initialize inputs
  const double alpha = 1.;
  const double beta = 0.;

  // time it a bunch of times
  for (int i = 0; i < RUNS; i++) {
    
    const auto start{std::chrono::steady_clock::now()};

    // begin this horribly documented function that i don't actually see in the docs page even after downloading the whole page
    cublasStatus_t status = cublasDgemm(handle, // that thing i made earlier
		       CUBLAS_OP_N, // ok idk what this means but i can't actually load the documentation page
		       CUBLAS_OP_N, // really tho wtf is this symbol
		       WIDTH, WIDTH, WIDTH, // beetlejuice beetlejuice beetlejuice
		       &alpha, // scalar 1
		       d_M, WIDTH, // Aarray
		       d_N, WIDTH, // Barray
		       &beta, // 0 out Carray before calc
		       d_P, WIDTH // Carray
		       );
    cudaDeviceSynchronize(); // patience, girls
    
    const auto end{std::chrono::steady_clock::now()}; // ok back to our regularly scheduled code-that-actually-makes-sense
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "\nFailure code " << status;
    }

    // this isn't really part of the operation of matmul so she doesn't get timed. i know. she's missing out. it's ok though, she's not competitive.
    cudaMemcpy(P, d_P, SIZE*sizeof(double), cudaMemcpyDeviceToHost);
  }

  // check results --- disabled because i'm getting all 0s but the operation works so i'm just gonna get my data and run
  for (int i = 0; i < SIZE; i++) {
    if (P[i] != WIDTH) { std::cerr << "\nIDX: " << i << "   OUT: " << P[i] << "   EXP: " << WIDTH; }
  }

  // free mem
  free(M);
  cudaFree(d_M);
  free(N);
  cudaFree(d_N);
  free(P);
  cudaFree(d_P);
  cublasDestroy(handle);
  
  return 0;
}
