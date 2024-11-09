#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iostream>

#define WIDTH 64  // Ends up being squared for the proper calculation (e.g. SIZE 4 means 4x4 matrices)
#define TILE_WIDTH 16
#define RUNS 100

// function to calculate the scan on GPU
__global__ void matmul(float *M, float *N, float *P, int width){
  int col = blockIdx.x*blockDim.x+threadIdx.x; // cols are .x in grid setup
  int row = blockIdx.y*blockDim.y+threadIdx.y; // rows are .y in grid setup

  if (row < width && col < width) { // inside range
    float sum = 0;
    for (int k=0; k < width; k++) {
      float a = M[ row*width + k ];
      float b = N[ k*width + col ];
      sum += a*b;
    }
    P[ row*width + col ] = sum;
  }
}

int main() {
  int SIZE = WIDTH*WIDTH; // compatibility with ported code from scan. since matrices are being initialized as 1D arrays, need a single size var. i'd prefer 2D. just sayin'
  std::cout << "\n" << SIZE << ","; // record the size of the run for data collection
  
  // allocate input and output arrays
  float *M, *d_M, *N, *d_N, *P, *d_P;
  M = (float*) malloc(SIZE*sizeof(float)); N = (float*) malloc(SIZE*sizeof(float)); P = (float*) malloc(SIZE*sizeof(float));
  cudaMalloc(&d_M, SIZE*sizeof(float)); cudaMalloc(&d_N, SIZE*sizeof(float)); cudaMalloc(&d_P, SIZE*sizeof(float));

  // initialize inputs
  for (int j = 0; j < SIZE; j++) {
    M[j] = 1;
    N[j] = 1;
  }
  cudaMemcpy(d_M, M, SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, N, SIZE*sizeof(float), cudaMemcpyHostToDevice);

  // initialize grid for indexing in matmul
  float* temp; temp = (float*) malloc(sizeof(float)); // don't ask why i do the things i do in code.
  temp[0] = ceil( (1.*WIDTH)/TILE_WIDTH ); // normal dereferencing for a pointer that is not an array.
  dim3 dimGrid(temp[0], temp[0], 1); // this line is :sparkle: magic :sparkle:
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  free(temp); // ...annndddd she's gone. out of sight, out of mind. just like she said. on the tag.

  // time it a bunch of times
  for (int i = 0; i < RUNS; i++) {
    
    const auto start{std::chrono::steady_clock::now()};
    matmul<<< dimGrid, dimBlock >>>(d_M, d_N, d_P, WIDTH);
    cudaDeviceSynchronize(); // patience, girls
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << elapsed.count() << "\n";

    // this isn't really part of the operation of matmul so she doesn't get timed. i know. she's missing out. it's ok though, she's not competitive.
    cudaMemcpy(P, d_P, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  }

  // check results
  for (int i = 0; i < SIZE; i++) {
    if (P[i] != WIDTH) { std::cerr << "IDX: " << i << "   OUT: " << P[i] << "   EXP: " << WIDTH << "\n"; }
  }

  // free mem
  free(M);
  cudaFree(d_M);
  free(N);
  cudaFree(d_N);
  free(P);
  cudaFree(d_P);

  return 0;
}
