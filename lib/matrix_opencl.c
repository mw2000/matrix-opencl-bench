#include <OpenCL/opencl.h>
#include <time.h>
#include "matrix_opencl.h"

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

    const char* kernelSource =
        "__kernel void mat_add(__global const float* A, __global const float* B, __global float* C) {  \n"
        "    int id = get_global_id(0); \n"
        "    C[id] = A[id] + B[id]; \n"
        "} \n";

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

    const char* kernelSource =
        "__kernel void mat_mul(__global const float* A, __global const float* B, __global float* C, int n) { \n"
        "    int row = get_global_id(0); \n"
        "    int col = get_global_id(1); \n"
        "    float sum = 0; \n"
        "    for (int k = 0; k < n; k++) { \n"
        "        sum += A[row * n + k] * B[k * n + col]; \n"
        "    } \n"
        "    C[row * n + col] = sum; \n"
        "} \n";

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