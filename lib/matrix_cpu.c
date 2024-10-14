#include "matrix_cpu.h"

// Function to perform matrix addition on CPU
void matrix_add_cpu(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// Function to perform matrix multiplication on CPU
// A and B are n x n matrices, C is the resulting n x n matrix
void matrix_multiply_cpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}