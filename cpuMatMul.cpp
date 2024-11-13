#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

#define WIDTH 1024  // Ends up being squared for the proper calculation (e.g. SIZE 4 means 4x4 matrices)
#define RUNS 100

// function to calculate the scan on GPU
void matmul(float *M, float *N, float *P, int width){
  for (int i=0; i < width; i++) {
    for (int j=0; j < width; j++) {
      float sum = 0;
      for (int k=0; k < width; k++) {
	float a = M[i*width +k];
	float b = N[k*width +j];
	sum += a*b;
      }
      P[i*width +j] = sum;
    }
  }
}

int main() {
  int SIZE = WIDTH*WIDTH; // compatibility with ported code from scan. since matrices are being initialized as 1D arrays, need a single size var. i'd prefer 2D. just sayin'
  std::cout << "\n" << SIZE; // record the size of the run for data collection

  // allocate input and output arrays
  float* M, *N, *P;
  M = (float*) malloc(SIZE*sizeof(float)); N = (float*) malloc(SIZE*sizeof(float)); P = (float*) malloc(SIZE*sizeof(float));
  
  
  for (int i = 0; i < RUNS; i++) {
    // initialize inputs
    for (int j = 0; j < SIZE; j++) {
      M[j] = 1;
      N[j] = 1;
    }
    
    const auto start{std::chrono::steady_clock::now()};
    matmul(M, N, P, WIDTH);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed{end - start};
    std::cout << "," << elapsed.count();
  }

  // check results
  for (int i = 0; i < SIZE; i++) {
    if (P[i] != WIDTH) { std::cerr << "IDX: " << i << "   OUT: " << P[i] << "   EXP: " << WIDTH << "\n"; }
  }

  // free mem
  free(M);
  free(N);
  free(P);

  return 0;
}
