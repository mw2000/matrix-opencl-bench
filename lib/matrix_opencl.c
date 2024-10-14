#include <OpenCL/opencl.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix_opencl.h"

char* read_kernel_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(1);  // Properly exit if the file can't be opened
    }

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);  // Get the file size
    if (size == 0) {
        fprintf(stderr, "Kernel file is empty: %s\n", filename);
        exit(1);  // Exit if the file is empty
    }

    rewind(file);  // Go back to the start of the file

    char* source = (char*)malloc(size + 1);  // Allocate memory for the source
    if (!source) {
        fprintf(stderr, "Failed to allocate memory for kernel source\n");
        fclose(file);  // Close the file before exiting
        exit(1);  // Properly handle memory allocation failure
    }

    // Read the file into memory
    fread(source, sizeof(char), size, file);
    source[size] = '\0';  // Null-terminate the kernel source

    fclose(file);  // Close the file after reading

    return source;
}


// Function to perform matrix addition using OpenCL
double matrix_add_opencl(const float* A, const float* B, float* C, int size) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem d_A = NULL, d_B = NULL, d_C = NULL;
    cl_int err;

    const char* kernelSource = read_kernel_file("lib/kernels/matrix_add.cl");

    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &err);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * size, NULL, &err);

    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, sizeof(float) * size, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, sizeof(float) * size, B, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    err |= clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "mat_add", &err);
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);

    size_t global_size = size;
    clock_t start = clock();
    err |= clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * size, C, 0, NULL, NULL);
    clock_t end = clock();

    double opencl_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return opencl_time_used;
}

// Function to perform matrix multiplication using OpenCL
double matrix_multiply_opencl(const float* A, const float* B, float* C, int n) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem d_A = NULL, d_B = NULL, d_C = NULL;
    cl_int err;

    const char* kernelSource = read_kernel_file("lib/kernels/matrix_mul.cl");

    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create memory buffers on the device
    d_A = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * n, NULL, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n * n, NULL, &err);
    d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n, NULL, &err);

    // Copy matrices A and B to device memory
    clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, sizeof(float) * n * n, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, sizeof(float) * n * n, B, 0, NULL, NULL);

    // Create and build the program from the kernel source
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    err |= clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "mat_mul", &err);

    // Set kernel arguments
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Define the global and local work size (2D grid for matrix multiplication)
    size_t global[2] = { n, n };

    // Start timing
    clock_t start = clock();
    
    // Execute the OpenCL kernel
    err |= clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);

    // Read the result back into C
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(float) * n * n, C, 0, NULL, NULL);
    
    // End timing
    clock_t end = clock();
    double opencl_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Clean up
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return opencl_time_used;
}