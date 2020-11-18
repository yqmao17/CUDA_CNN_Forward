#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8
#define BLOCK_SIZE 512

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int b = blockIdx.x;
    int m = blockIdx.y;
    int W_grid = ceil(W_out * 1.0/ TILE_WIDTH);
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    float accu = 0;
    if(w < (W_out) && h < (H_out)){
        for (int c = 0; c < C; c++){
            for (int p = 0; p < K; p++){
                for (int q = 0; q < K; q++){
                    accu += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = accu;
    }

#undef y4d
#undef x4d
#undef k4d
}

// __global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns){
//     //@@ Insert code to implement matrix multiplication here
//     //@@ You have to use shared memory for this MP
//     // set blockDim.x and y = TILE_W
//     __shared__ float tile_a[TILE_W][TILE_W];
//     __shared__ float tile_b[TILE_W][TILE_W];

//     int bx_idx = blockIdx.x;
//     int by_idx = blockIdx.y;
//     int tx_idx = threadIdx.x;
//     int ty_idx = threadIdx.y;
    
//     int row = by_idx * TILE_W + ty_idx;
//     int col = bx_idx * TILE_W + tx_idx;
//     float accu = 0;
//     for(int n = 0; n < ceil(numAColumns * 1.0 / TILE_W); n++){
//         if(row < numARows && n * TILE_W + tx_idx < numAColumns){
//         tile_a[ty_idx][tx_idx] = A[row * numAColumns + n * TILE_W + tx_idx];
//         }else{
//         tile_a[ty_idx][tx_idx] = 0;
//         }
//         if(col < numBColumns && n * TILE_W + ty_idx < numBRows){
//         tile_b[ty_idx][tx_idx] = B[(n * TILE_W + ty_idx) * numBColumns + col];
//         }else{
//         tile_b[ty_idx][tx_idx] = 0;
//         }
//         __syncthreads();
//         for(int i = 0; i < TILE_W; i++){
//         accu += tile_a[ty_idx][i] * tile_b[i][tx_idx];
//         }
//         __syncthreads();
//     }
//     if(row < numCRows && col < numCColumns){
//         C[row * numCColumns + col] = accu;
//     }
// }

// __global__ void unroll_kernel(int C, int H, int W, int K, float *X, float *X_unroll){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int H_out = H - K + 1;
//     int W_out = W - K + 1;
//     int W_unroll = H_out * W_out;
//     if(idx < C * W_unroll){
//         int c = idx / W_unroll;
//         int s = idx % W_unroll;
//         int h_out = s / W_out;
//         int w_out = s % W_out;
//         int w_base = c * K * K;
//         int w_unroll = h_out * W_out + w_out;
//         for(int p = 0; p < K; p++){
//             for(int q = 0; q < K; q++){
//                 int h_unroll = w_base + p * K + q;
//                 X_unroll[h_unroll * W_unroll + w_unroll] = X[c * H * W + (h_out + p) * W + w_out + q];
//             }
//         }
//     }
// }

// void unroll_to_gpu(int C, int H, int W, int K, float *X, float *X_unroll){
//     int W_out = W - K + 1;
//     int H_out = H - K + 1;
//     dim3 unroll_griddim(ceil(C * H_out * W_out * 1.0 / BLOCK_SIZE), 1, 1);
//     dim3 unroll_blockdim(BLOCK_SIZE, 1, 1);
//     unroll_kernel<<<unroll_griddim, unroll_blockdim>>>(C, H, W, K, X, X_unroll);
// }

// void matrixMultiply(float *KK, float *X_unrolled, float *Y, int H_unroll, int M, int W_unroll){
//     int numCColumns = W_unroll;
//     int numCRows = M;
//     int numAColumns = H_unroll;
//     int numARows = numCRows;
//     int numBRows = numAColumns;
//     int numBColumns = numCColumns;
//     dim3 mm_griddim(ceil(1.0 * W_unroll / TILE_WIDTH),  ceil(1.0 *  M / TILE_WIDTH), 1);
//     dim3 mm_blockdim(TILE_WIDTH, TILE_WIDTH, 1);
//     matrixMultiplyShared<<<mm_griddim, mm_blockdim>>>(KK, X_unrolled, Y, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
// }




// __host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     // Declare relevant device pointers
//     float *device_y;
//     float *device_x;
//     float *device_k;
//     float *X_unrolled;
//     // Allocate memory and copy over the relevant data structures to the GPU
//     cudaMalloc((void **) &device_y, B*M*(H - K + 1)*(W - K + 1)*sizeof(float));
//     cudaMalloc((void **) &device_x, B*C*H*W*sizeof(float));
//     cudaMalloc((void **) &device_k, C*M*K*K*sizeof(float));
//     cudaMemcpy(device_x, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(device_k, host_k, C*M*K*K*sizeof(float), cudaMemcpyHostToDevice);
//     // Set the kernel dimensions and call the kernel
//     int H_out = H - K + 1;
//     int W_out = W - K + 1;
//     int W_unroll = H_out * W_out;
//     int H_unroll = C * K * K;
//     cudaMalloc((void **) &X_unrolled, W_unroll * H_unroll * sizeof(float));
//     for(int b = 0; b < B; b++){ 
//         unroll_to_gpu(C, H, W, K, device_x+b*C*H*W, X_unrolled);
//         matrixMultiply(device_k, X_unrolled, device_y+b*M*H_out*W_out, H_unroll, M, W_unroll);
//     }
//     cudaDeviceSynchronize();
//     // Copy the output back to host
//     cudaMemcpy(host_y, device_y, B*M*(H - K + 1)*(W - K + 1)*sizeof(float), cudaMemcpyDeviceToHost);
//     // Free device memory
//     cudaFree(device_y);
//     cudaFree(device_x);
//     cudaFree(device_k);
//     cudaFree(X_unrolled);
//     // Useful snippet for error checking
//     // cudaError_t error = cudaGetLastError();
//     // if(error != cudaSuccess)
//     // {
//     //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
//     //     exit(-1);
//     // }
// }

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *device_y;
    float *device_x;
    float *device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float));
    cudaMalloc((void **) &device_x, B*C*H*W*sizeof(float));
    cudaMalloc((void **) &device_k, C*M*K*K*sizeof(float));
    cudaMemcpy(device_x, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, C*M*K*K*sizeof(float), cudaMemcpyHostToDevice);
    // Set the kernel dimensions and call the kernel
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_grid = ceil(W_out * 1.0 / TILE_WIDTH);
    int H_grid = ceil(H_out * 1.0 / TILE_WIDTH);
    int Z = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);
    conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    cudaDeviceSynchronize();
    // Copy the output back to host
    cudaMemcpy(host_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
