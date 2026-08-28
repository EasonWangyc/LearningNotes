/*
矩阵乘法是整个深度学习和AI Infra的绝对算力核心。

假设需要计算C = A × B，在朴素（Naive）矩阵乘法中，计算 $C$ 的每一个元素都需要从全局显存（Global Memory）把 $A$ 的整行和 $B$ 的整列读一遍，访存次数为 $2N^3$。

Tile Gemm利用片上的 共享内存（Shared Memory / SRAM）。
将矩阵切分为 TILE_SIZE × TILE_SIZE 的小块（比如 32 × 32）。同一个 Thread Block 内的所有线程协同合作，先把一个小块的数据从 Global Memory 搬到 Shared Memory 中。随后每个线程直接从片上 Shared Memory 中高速读取数据进行内积累加。全局显存的访存量直接降低为原本的 1/TILE_SIZE，计算吞吐大幅飙升！
*/
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// ==========================================
// 1. Tiled GEMM 核函数 (利用 Shared Memory)
// ==========================================
// 计算 C (M x N) = A (M x K) * B (K x N)
__global__ void tiledMatrixMul(const float* A, const float* B, float *C, int M, int N, int K){
    // 生命共享内存，由当前block内的所有线程共享
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    /*
      blockIdx.x=0     blockIdx.x=1     ...   (C 的列方向，对应 N)
        ┌──────────────┬──────────────┐
        │ 32×32 的 tile │              │
blockIdx.y=0  (0,0)   (0,1)
        ├──────────────┼──────────────┤
blockIdx.y=1  (1,0)   (1,1)
        │              │              │
        └──────────────┴──────────────┘
          C 的行方向，对应 M
    */
    
    // GPU 是SIMT（单指令多线程）模型
    // 硬件发给每个线程一个不同的常量
    // 计算当前线程在block内的局部坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算当前线程对应输出矩阵C的全局行列索引，列对应x，行对应y
    int row = blockIdx.y * TILE_SIZE + ty;  // C的第几行
    int col = blockIdx.x * TILE_SIZE + tx;  // C的第几列

    float acc = 0.0f;

    // 沿k维度分阶段（Phase）迭代加载Tile
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for(int t = 0; t < numTiles; t++){
        // 1.协同搬运，每个线程从Global memory中搬运一个元素到s_a和s_b
        int a_col = t * TILE_SIZE + tx; //A的列，第t个tile内偏移tx
        if(row < M && a_col < K){
            s_a[ty][tx] = A[row * K + a_col];
        }else{
            s_a[ty][tx] = 0.0f; //边界检查
        }
        int b_row = t * TILE_SIZE + ty; //B的行，第t个tile内偏移ty
        if(b_row < K && col < N){
            s_b[ty][tx] = B[b_row * N + col];
        }else{
            s_b[ty][tx] = 0.0f; //边界检查
        }
        // 2.块内同步，确保当前Tile的数据已全部载入shared memory
        // 执行同步：当某个线程块中的某个线程执行到 __syncthreads() 时，它会暂停运行，直到该线程块内的所有其他线程也都到达这个同步点，大家才能一起向下执行
        // 第一个syncthreads()保证所有搬运结束后才读
        __syncthreads();
        // 3.片上高速计算，在shared memory中做内积累加
        #pragma unroll
        for(int j = 0; j < TILE_SIZE; j++){
            // 对于单个block，Σ_{k=0}^{31} A[ty][j] * B[j][tx]
            acc += s_a[ty][j] * s_b[j][tx];
        }
        // 第二个syncthreads()保证所有都计算完成再开启下一轮
        __syncthreads();
    }
    if(row < M && col < N){
        C[row * N + col] = acc;
    }
}

// ==========================================
// 2. CPU 朴素矩阵乘法 (用于结果精度对齐)
// ==========================================
void cpuMatrixMul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ==========================================
// 3. Host 主程序与性能基准测试
// ==========================================
int main(){
    // 矩阵规模定义：M=1024, N=1024, K=1024
    int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // host内存分配
    float *h_A = (float *) malloc(bytes_A);
    float *h_B = (float *) malloc(bytes_B);
    float *h_C = (float *) malloc(bytes_C);
    float *h_C_ref = (float *) malloc(bytes_C);
    // 初始化随机数据
    for(int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 100) / 100.0f;
    for(int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 100) / 100.0f;
    // device显存分配与拷贝，声明指针
    float *d_A, *d_B, *d_C;
    // cudaMalloc(void** devPtr, size_t size)
    // 第一个参数为指针的地址，写入新分配的显存地址
    cudaMalloc(&d_A, bytes_A);  // 出参，把分配的显存地址写进d_A
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    // 设置 2D 线程网格结构
    dim3 blockDim(TILE_SIZE, TILE_SIZE); // 32 x 32 = 1024 线程/Block
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // 性能计时 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // <<<gridDim, blockDim>>> 内核启动配置
    // CPU 指定 GPU：开多少个 block、每个 block 多少线程
    // Warm-up 预热
    tiledMatrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // 正式执行测速
    cudaEventRecord(start);
    tiledMatrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // 拷贝结果回 Host
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);

    // 验证部分数据的正确性 (防止 CPU 计算 1024x1024 太慢，随机抽检)
    std::cout << "正在校验精度..." << std::endl;
    float max_diff = 0.0f;
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            float ref = 0.0f;
            for (int k = 0; k < K; ++k) ref += h_A[i * K + k] * h_B[k * N + j];
            max_diff = std::max(max_diff, std::abs(h_C[i * N + j] - ref));
        }
    }

    // 吞吐量计算 (TFLOPS)
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / (ms / 1000.0)) / 1e12;

    std::cout << "---------------------------------------" << std::endl;
    std::cout << "矩阵尺寸: [" << M << "x" << K << "] * [" << K << "x" << N << "]" << std::endl;
    std::cout << "最大误差: " << max_diff << std::endl;
    std::cout << "耗时: " << ms << " ms" << std::endl;
    std::cout << "算力吞吐: " << tflops << " TFLOPS" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    if (max_diff < 1e-3) {
        std::cout << "✅ Tiled GEMM 计算成功且精度一致！" << std::endl;
    }

    // 释放资源
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}

// nvcc -O3 tiled_gemm.cu -o tiled_gemm
// ./tiled_gemm