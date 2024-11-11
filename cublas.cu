#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define WIDTH 4  // Ends up being squared for the proper calculation (e.g. SIZE 4 means 4x4 matrices)
#define TILE_WIDTH 1
#define RUNS 1


int main() {
  const int SIZE = WIDTH*WIDTH; // compatibility with ported code from scan. since matrices are being initialized as 1D arrays, need a single size var. i'd prefer 2D. just sayin'
  std::cout << "\n" << SIZE; // record the size of the run for data collection

  // allocate input and output arrays
  double *M, *N, *P;
  cudaMallocManaged(&M, SIZE*sizeof(double)); cudaMallocManaged(&N, SIZE*sizeof(double)); cudaMallocManaged(&P, SIZE*sizeof(double));

  // initialize inputs
  for (int i = 0; i < SIZE; i++) {
    M[i] = 1;
    N[i] = 1;
  }
    
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
    cublasDgemmBatched(handle, // that thing i made earlier
		       CUBLAS_OP_N, // ok idk what this means but i can't actually load the documentation page
		       CUBLAS_OP_N, // really tho wtf is this symbol
		       WIDTH, WIDTH, WIDTH, // beetlejuice beetlejuice beetlejuice
		       &alpha, // scalar 1
		       &M, WIDTH, // Aarray
		       &N, WIDTH, // Barray
		       &beta, // 0 out Carray before calc
		       &P, WIDTH, // Carray
		       1 // documentation is unclear whether this is supposed to be WIDTH or 3*WIDTH or 30 billion or what...
		       );
    cudaDeviceSynchronize(); // patience, girls
    
    const auto end{std::chrono::steady_clock::now()}; // ok back to our regularly scheduled code-that-actually-makes-sense
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();
  }

  // check results
  for (int i = 0; i < SIZE; i++) {
    if (P[i] != WIDTH) { std::cerr << "IDX: " << i << "   OUT: " << P[i] << "   EXP: " << WIDTH << "\n"; }
  }

  cudaFree(M);
  cudaFree(N);
  cudaFree(P);

  return 0;
}
