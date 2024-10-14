# OpenCL Matrix Multiplication and Addition Benchmarking

This project benchmarks matrix addition and matrix multiplication using OpenCL, comparing their performance to CPU implementations. OpenCL (Open Computing Language) is a cross-platform framework that enables execution of programs on heterogeneous systems such as CPUs, GPUs, and other processors.

## Table of Contents
- [Introduction](#introduction)
- [Benchmarks](#benchmarks)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Matrix operations like addition and multiplication are fundamental to many computational tasks. This project showcases the performance benefits of using OpenCL for matrix operations compared to traditional CPU implementations. It demonstrates how OpenCL can leverage the power of modern GPUs to accelerate these operations, particularly for large matrices.

The project includes:
- Matrix addition using both CPU and OpenCL.
- Matrix multiplication using both CPU and OpenCL.
- A comparison of the time taken by the CPU and OpenCL for varying matrix sizes.

## Benchmarks on MacBook M1 Pro

The following results were obtained on a MacBook M1 Pro. Times are shown for both CPU and OpenCL execution across different matrix sizes.

### Matrix Addition Benchmark

| Matrix Size | CPU Time (seconds) | OpenCL Time (seconds) |
|-------------|-------------------|----------------------|
| 10 x 10     | 0.000002          | 0.000200             |
| 100 x 100   | 0.000016          | 0.000165             |
| 1000 x 1000 | 0.000553          | 0.000628             |
| 10000 x 10000| 0.053535         | 0.037118             |

### Matrix Multiplication Benchmark

| Matrix Size | CPU Time (seconds) | OpenCL Time (seconds) |
|-------------|-------------------|----------------------|
| 10 x 10     | 0.000008          | 0.000247             |
| 100 x 100   | 0.001113          | 0.000098             |
| 1000 x 1000 | 1.212626          | 0.000517             |
| 2000 x 2000 | 10.351148         | 0.001509             |

As the matrix size increases, the performance advantage of using OpenCL becomes more apparent, especially for large matrices where GPU acceleration really shines.

## Installation

### Prerequisites

You will need the following to build and run the project:
- **OpenCL SDK** (already available on macOS)
- **gcc** compiler

### Installing OpenCL

Depending on your platform, you may need to install OpenCL drivers and libraries:

1. **Windows:**
   - Download and install the OpenCL SDK from your GPU vendor (NVIDIA, AMD, Intel).

2. **Linux:**
   - Install OpenCL headers and libraries with the following command:
     ```sh
     sudo apt-get install ocl-icd-opencl-dev
     ```

3. **macOS:**
   - OpenCL comes pre-installed on macOS, so no additional installation is required.

### Building the Project

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/matrix-opencl-bench.git
    cd matrix-opencl-bench
    ```

2. Build the project using the Makefile:
    ```sh
    make
    ```

## Usage

To run the benchmark after building:

```sh
make run
```

This will execute both the matrix addition and matrix multiplication benchmarks, and the results will be displayed in the terminal.

You can also clean up the build artifacts using:

```sh
make clean
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

