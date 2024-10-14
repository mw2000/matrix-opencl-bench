#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix_cpu.h"
#include "matrix_opencl.h"

#define MATRIX_SIZE_10 10 // 10x10 matrix
#define MATRIX_SIZE_100 100 // 100x100 matrix
#define MATRIX_SIZE_1000 1000 // 1000x1000 matrix
#define MATRIX_SIZE_2000 2000 // 10000x10000 matrix
#define MATRIX_SIZE_10000 10000 // 10000x10000 matrix

// Function to test both CPU and OpenCL implementations
void test_matrix_addition(int matrix_length) {
    int matrix_size = matrix_length * matrix_length;

    float* A = (float*) malloc(matrix_size * sizeof(float));
    float* B = (float*) malloc(matrix_size * sizeof(float));
    float* C_cpu = (float*) malloc(matrix_size * sizeof(float));
    float* C_opencl = (float*) malloc(matrix_size * sizeof(float));

    for (int i = 0; i < matrix_size; i++) {
        A[i] = (float)(i + 1);
        B[i] = (float)(matrix_size - i);
    }

    // CPU Timing
    clock_t start = clock();
    matrix_add_cpu(A, B, C_cpu, matrix_size);
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // OpenCL Timing
    double opencl_time_used = matrix_add_opencl(A, B, C_opencl, matrix_size);

    // Print the results in a table-like format
    printf("| %-11d | %-17f | %-20f |\n", (int) sqrt(matrix_size), cpu_time_used, opencl_time_used);

    free(A);
    free(B);
    free(C_cpu);
    free(C_opencl);
}

// Function to test both CPU and OpenCL implementations
void test_matrix_multiplication(int matrix_length) {
    int matrix_size = matrix_length * matrix_length;

    float* A = (float*) malloc(matrix_size * sizeof(float));
    float* B = (float*) malloc(matrix_size * sizeof(float));
    float* C_cpu = (float*) malloc(matrix_size * sizeof(float));
    float* C_opencl = (float*) malloc(matrix_size * sizeof(float));

    for (int i = 0; i < matrix_size; i++) {
        A[i] = (float)(i + 1);
        B[i] = (float)(matrix_size - i);
    }

    // CPU Timing
    clock_t start = clock();
    matrix_multiply_cpu(A, B, C_cpu, matrix_length);
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // OpenCL Timing
    double opencl_time_used = matrix_multiply_opencl(A, B, C_opencl, matrix_length);

    // Print the results in a table-like format
    printf("| %-11d | %-17f | %-20f |\n", matrix_length, cpu_time_used, opencl_time_used);

    free(A);
    free(B);
    free(C_cpu);
    free(C_opencl);
}

int main() {
    printf("Matrix Addition\n");
    printf("|-------------|-------------------|----------------------|\n");
    printf("| Matrix Size | CPU Time (secs)   | OpenCL Time (secs)   |\n");
    printf("|-------------|-------------------|----------------------|\n");

    test_matrix_addition(MATRIX_SIZE_10);    // 10x10 matrix
    test_matrix_addition(MATRIX_SIZE_100);   // 100x100 matrix
    test_matrix_addition(MATRIX_SIZE_1000);  // 1000x1000 matrix
    test_matrix_addition(MATRIX_SIZE_10000); // 10000x10000 matrix
    printf("|--------------------------------------------------------|\n");

    printf("\n\n\n");

    printf("Matrix Multiplication\n");
    printf("|-------------|-------------------|----------------------|\n");
    printf("| Matrix Size | CPU Time (secs)   | OpenCL Time (secs)   |\n");
    printf("|-------------|-------------------|----------------------|\n");

    test_matrix_multiplication(MATRIX_SIZE_10);   // 10x10 matrix
    test_matrix_multiplication(MATRIX_SIZE_100);  // 100x100 matrix
    test_matrix_multiplication(MATRIX_SIZE_1000); // 1000x1000 matrix
    test_matrix_multiplication(MATRIX_SIZE_2000); // 2000x2000 matrix

    printf("|--------------------------------------------------------|\n"); 
    return 0;
}
