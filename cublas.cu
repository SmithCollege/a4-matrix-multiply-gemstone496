#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define WIDTH 64  // Ends up being squared for the proper calculation (e.g. SIZE 4 means 4x4 matrices)
#define TILE_WIDTH 16
#define RUNS 1


int main() {
  const int SIZE = WIDTH*WIDTH; // compatibility with ported code from scan. since matrices are being initialized as 1D arrays, need a single size var. i'd prefer 2D. just sayin'
  std::cout << "\n" << SIZE; // record the size of the run for data collection
  
  // allocate input and output arrays
  const int MEM_SIZE = sizeof(double*) * WIDTH;
  const int MEM_P2 = sizeof(double) * WIDTH;
  
  double **M, **d_M, **N, **d_N, **P, **d_P;
  M = (double**) malloc(MEM_SIZE); N = (double**) malloc(MEM_SIZE); P = (double**) malloc(MEM_SIZE);
  cudaMalloc(&d_M, MEM_SIZE); cudaMalloc(&d_N, MEM_SIZE); cudaMalloc(&d_P, MEM_SIZE);

  for (int i=0; i < WIDTH; i++) {
    M[i] = (double*)  malloc(MEM_P2); N[i] = (double*) malloc(MEM_P2); P[i] = (double*) malloc(MEM_P2);
    cudaMalloc(&(d_M[i]), MEM_P2); cudaMalloc(&(d_N[i]), MEM_P2); cudaMalloc(&(d_P[i]), MEM_P2);
  }
    
  // initialize handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // initialize inputs
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      M[i][j] = 1;
      N[i][j] = 1;
    }
    cudaMemcpy((void*) d_M[i], (void*) M[i], WIDTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_N[i], (void*) N[i], WIDTH*sizeof(float), cudaMemcpyHostToDevice);
  }
  const double alpha = 1.;
  const double beta = 0.;

  // time it a bunch of times
  for (int i = 0; i < RUNS; i++) {
    
    const auto start{std::chrono::steady_clock::now()};
    // begin this horribly documented function that i don't actually see in the docs page even after downloading the whole page
    cublasDgemmBatched(handle, // that thing i made earlier
		       CUBLAS_OP_N, // ok idk what this means but i can't actually load the documentation page
		       CUBLAS_OP_N, // really tho wtf is this symbol
		       WIDTH, WIDTH, WIDTH, // beetlejuice beetlejuice beetlejuice
		       &alpha, // scalar 1
		       d_M, WIDTH, // Aarray
		       d_N, WIDTH, // Barray
		       &beta, // 0 out Carray before calc
		       d_P, WIDTH, // Carray
		       WIDTH // documentation is unclear whether this is supposed to be WIDTH or 3*WIDTH or 30 billion or what...
		       );
    cudaDeviceSynchronize(); // patience, girls
    
    const auto end{std::chrono::steady_clock::now()}; // ok back to our regularly scheduled code-that-actually-makes-sense
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();
  }

  // check results
  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      if (P[i][j] != WIDTH) { std::cerr << "ROW: " << i << "   COL: " << j << "   OUT: " << P[i] << "   EXP: " << WIDTH << "\n"; }
    }
  }

  // free mem
  for (int i = 0; i < WIDTH; i++) {
    free((void*)M[i]); free((void*)N[i]); free((void*)P[i]);
    cudaFree((void*)d_M[i]); cudaFree((void*)d_N[i]); cudaFree((void*)d_P[i]);
  }
  free((void*)M);
  cudaFree((void*)d_M);
  free((void*)N);
  cudaFree((void*)d_N);
  free((void*)P);
  cudaFree((void*)d_P);

  return 0;
}
