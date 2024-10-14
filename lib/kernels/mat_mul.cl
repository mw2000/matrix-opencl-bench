__kernel void mat_mul(__global const float* A, __global const float* B, __global float* C, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0;
    for (int k = 0; k < n; k++) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}