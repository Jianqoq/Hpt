#include <stdio.h>

#define type float
#define type_vec float4

extern "C" __global__ void matmul_naive(type *a, type *b, type *out, size_t m, size_t n, size_t k, size_t batch_size)
{
    if (blockIdx.z >= batch_size)
        return;

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
        return;

    type value = 0.0f;
    for (size_t i = 0; i < k; i++)
    {
        value += a[blockIdx.z * m * k + row * k + i] * b[blockIdx.z * k * n + i * n + col];
    }

    out[blockIdx.z * m * n + row * n + col] = value;
}

#define TILE_SIZE 16

__device__ void mm4x4(type a[4], type b[4], type out[4][4])
{
#pragma unroll
    for (size_t i = 0; i < 4; i++)
    {
        out[i][0] += a[i] * b[0];
        out[i][1] += a[i] * b[1];
        out[i][2] += a[i] * b[2];
        out[i][3] += a[i] * b[3];
    }
}

extern "C" __global__ void matmul_blocked(type *a, type *b, type *out, size_t m, size_t n, size_t k, size_t batch_size)
{

    __shared__ type a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ type b_tile[TILE_SIZE][TILE_SIZE];

    if (blockIdx.z >= batch_size)
        return;

    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n)
        return;

    type value = 0.0f;
    size_t end = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t i = 0; i < end; i++)
    {
        if (i * TILE_SIZE + threadIdx.x < k)
        {
            a_tile[threadIdx.y][threadIdx.x] = a[blockIdx.z * m * k + row * k + i * TILE_SIZE + threadIdx.x];
        }
        else
        {
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (i * TILE_SIZE + threadIdx.y < k)
        {
            b_tile[threadIdx.y][threadIdx.x] = b[blockIdx.z * k * n + (i * TILE_SIZE + threadIdx.y) * n + col];
        }
        else
        {
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (size_t j = 0; j < TILE_SIZE; j++)
        {
            value += a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x];
        }
        __syncthreads();
    }

    out[blockIdx.z * m * n + row * n + col] = value;
}

template <size_t M, size_t N, size_t K, size_t TileSize>
__device__ void load_to_smem(
    const type *A, const type *B,
    size_t j, size_t m, size_t n, size_t k,
    size_t row, size_t col,
    type a_tile[M][K], type b_tile[K][N])
{
// 加载 A 块到共享内存
#pragma unroll
    for (int i = 0; i < TileSize; i++)
    {
        size_t a_row = row + threadIdx.y * TileSize + i;
        size_t a_col = j + threadIdx.x;
        if (a_row < m && a_col < k)
        {
            a_tile[threadIdx.y * TileSize + i][threadIdx.x] = A[blockIdx.z * m * k + a_row * k + a_col];
        }
        else
        {
            a_tile[threadIdx.y * TileSize + i][threadIdx.x] = 0.0f;
        }
    }

// 加载 B 块到共享内存
#pragma unroll
    for (int i = 0; i < TileSize; i++)
    {
        size_t b_row = j + threadIdx.y * TileSize + i;
        size_t b_col = col + threadIdx.x;
        if (b_row < k && b_col < n)
        {
            b_tile[threadIdx.y * TileSize + i][threadIdx.x] = B[blockIdx.z * k * n + b_row * n + b_col];
        }
        else
        {
            b_tile[threadIdx.y * TileSize + i][threadIdx.x] = 0.0f;
        }
    }
}

// define tile and thread processing size
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define THREAD_BLOCK_X 16
#define THREAD_BLOCK_Y 16
#define VEC_SIZE 4

template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileM, size_t TileK>
__device__ void load_gmem_to_next_a_reg(const T *A, T *next_data_a, size_t txa, size_t tya, size_t k, const size_t tile_idx)
{
    const float *pA = (A + (txa * ReadPerThread + tile_idx) + (blockIdx.y * TileM + tya) * k);
#pragma unroll
    for (int i = 0; i < ReadPerThread; i++)
    {
        next_data_a[i] = pA[i];
    }
}

template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileN, size_t TileK>
__device__ void load_gmem_to_next_b_reg(const T *B, T *next_data_b, size_t txb, size_t tyb, size_t n, const size_t tile_idx)
{
    const float *pB = (B + txb * ReadPerThread + blockIdx.x * TileN + tyb * n + tile_idx * n);

    for (int i = 0; i < ReadPerThread; i++)
    {
        next_data_b[i] = pB[i];
    }
}

template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileK, size_t smem_size>
__device__ void store_next_a_data_to_smem(T next_data[ReadPerThread], T smem[smem_size], size_t txa, size_t tya)
{
    for (int i = 0; i < ReadPerThread; i++)
    {
        smem[(txa * ReadPerThread + i) * TILE_M + tya] = next_data[i];
    }
}

template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileN, size_t smem_size>
__device__ void store_next_b_data_to_smem(T next_data[ReadPerThread], T smem[smem_size], size_t txb, size_t tyb)
{
    for (int i = 0; i < ReadPerThread; i++)
    {
        smem[txb * ReadPerThread + i + tyb * TileN] = next_data[i];
    }
}

template <typename T, size_t smem_size, size_t ReadPerThread, size_t TileM>
__device__ void load_smem_to_a_reg(T smem[smem_size], T reg[ReadPerThread], size_t tx, size_t ty, size_t j)
{
    for (int i = 0; i < ReadPerThread; i++)
    {
        reg[i] = smem[ty * ReadPerThread + i + TileM * j];
    }
}

template <typename T, size_t smem_size, size_t ReadPerThread, size_t TileN>
__device__ void load_smem_to_b_reg(T smem[smem_size], T reg[ReadPerThread], size_t tx, size_t ty, size_t j)
{
    for (int i = 0; i < ReadPerThread; i++)
    {
        reg[i] = smem[tx * ReadPerThread + i + TileN * j];
    }
}

template <typename T, size_t AReadPerThread, size_t BReadPerThread>
__device__ void mma(T a[AReadPerThread], T b[BReadPerThread], T c[AReadPerThread][BReadPerThread])
{
#pragma unroll
    for (size_t i = 0; i < AReadPerThread; ++i)
    {
#pragma unroll
        for (size_t j = 0; j < BReadPerThread; ++j)
        {
            c[i][j] += a[i] * b[j];
        }
    }
}

__device__ int lock = 0;
// 获取锁
__device__ void acquire_lock(int *lock)
{
    while (atomicCAS(lock, 0, 1) != 0)
    {
        // 自旋等待锁释放
    }
}

// 释放锁
__device__ void release_lock(int *lock)
{
    atomicExch(lock, 0);
}

// matrix multiplication kernel
extern "C" __global__ void matmul_blocked2(
    const type *A, const type *B, type *C,
    size_t m, size_t n, size_t k, size_t batch_size)
{
    // define shared memory
    __shared__ type a_tile[2][TILE_K * TILE_M];
    __shared__ type b_tile[2][TILE_K * TILE_N];

    constexpr const size_t a_read_per_thread = TILE_M * TILE_K / (THREAD_BLOCK_X * THREAD_BLOCK_Y);
    constexpr const size_t b_read_per_thread = TILE_N * TILE_K / (THREAD_BLOCK_X * THREAD_BLOCK_Y);

    static_assert(TILE_M % a_read_per_thread == 0, "a_read_per_thread must be multiple of TILE_M");
    static_assert(TILE_K % b_read_per_thread == 0, "b_read_per_thread must be multiple of TILE_K");
    static_assert(a_read_per_thread < TILE_K, "a_read_per_thread must be less than TILE_K");
    static_assert(b_read_per_thread < TILE_K, "b_read_per_thread must be less than TILE_K");

    type next_data_a[a_read_per_thread];
    type next_data_b[b_read_per_thread];
    type a_reg[2][a_read_per_thread]; // must be multiple of VEC_SIZE
    type b_reg[2][b_read_per_thread]; // must be multiple of VEC_SIZE

    // // initialize 4x4 accumulator
    type c_reg[a_read_per_thread][b_read_per_thread] = {{0.0f}};

    size_t txa = threadIdx.x % (TILE_K / a_read_per_thread);
    size_t tya = threadIdx.x / (TILE_K / a_read_per_thread);

    size_t txb = threadIdx.x % (TILE_N / b_read_per_thread);
    size_t tyb = threadIdx.x / (TILE_N / b_read_per_thread);
    // load data from global memory to next_data_regs, each next_data_a store TILE_K elements

    load_gmem_to_next_a_reg<type, VEC_SIZE, a_read_per_thread, TILE_M, TILE_K>(A, next_data_a, txa, tya, k, 0);
    load_gmem_to_next_b_reg<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K>(B, next_data_b, txb, tyb, n, 0);

    int global_idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * (gridDim.x * blockDim.x);

    store_next_a_data_to_smem<type, VEC_SIZE, a_read_per_thread, TILE_K, TILE_K * TILE_M>(next_data_a, a_tile[0], txa, tya);
    store_next_b_data_to_smem<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K * TILE_N>(next_data_b, b_tile[0], txb, tyb);
    size_t tx = threadIdx.x % THREAD_BLOCK_X;
    size_t ty = threadIdx.x / THREAD_BLOCK_X;

    __syncthreads();

    load_smem_to_a_reg<type, TILE_K * TILE_M, a_read_per_thread, TILE_M>(a_tile[0], a_reg[0], tx, ty, 0);
    load_smem_to_b_reg<type, TILE_K * TILE_N, b_read_per_thread, TILE_N>(b_tile[0], b_reg[0], tx, ty, 0);

    // acquire_lock(&lock);
    // if ((threadIdx.x == 0) && blockIdx.x == 1 && blockIdx.y == 0)
    // {
    //     printf("global_idx = %d, A: [", global_idx);
    //     for (int i = 0; i < a_read_per_thread; i++)
    //     {
    //         printf("%f, ", next_data_a[i]);
    //     }
    //     printf("], B: [");
    //     for (int i = 0; i < b_read_per_thread; i++)
    //     {
    //         printf("%f, ", next_data_b[i]);
    //     }
    //     printf("], (%llu, %llu)\n", tx, ty);
    //     // printf("a_tile[0]:\n");
    //     // for (int i = 0; i < TILE_K; i++)
    //     // {
    //     //     for (int j = 0; j < TILE_M; j++)
    //     //     {
    //     //         printf("%f, ", a_tile[0][i * TILE_M + j]);
    //     //     }
    //     //     printf("\n");
    //     // }
    //     // printf("\n");
    //     // printf("b_tile[0]:\n");
    //     // for (int i = 0; i < TILE_K; i++)
    //     // {
    //     //     for (int j = 0; j < TILE_N; j++)
    //     //     {
    //     //         printf("%f, ", b_tile[0][i * TILE_N + j]);
    //     //     }
    //     //     printf("\n");
    //     // }
    //     // printf("\n");
    //     printf("a_reg[0]:\n");
    //     for (int i = 0; i < a_read_per_thread; i++)
    //     {
    //         printf("%f, ", a_reg[0][i]);
    //     }
    //     printf("\n");
    //     // printf("b_reg[0]:\n");
    //     // for (int i = 0; i < b_read_per_thread; i++)
    //     // {
    //     //     printf("%f, ", b_reg[0][i]);
    //     // }
    //     // printf("\n");
    // }
    // release_lock(&lock);

    int i = 0;
    int write_stage_idx = 1;
    do
    {
        i += TILE_K;

        load_gmem_to_next_a_reg<type, VEC_SIZE, a_read_per_thread, TILE_M, TILE_K>(A, next_data_a, txa, tya, k, i);
        load_gmem_to_next_b_reg<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K>(B, next_data_b, txb, tyb, n, i);

        int load_stage_idx = write_stage_idx ^ 1;

        for (int j = 0; j < TILE_K - 1; j++)
        {
            load_smem_to_a_reg<type, TILE_K * TILE_M, a_read_per_thread, TILE_M>(a_tile[load_stage_idx], a_reg[(j + 1) % 2], tx, ty, j + 1);
            load_smem_to_b_reg<type, TILE_K * TILE_N, b_read_per_thread, TILE_N>(b_tile[load_stage_idx], b_reg[(j + 1) % 2], tx, ty, j + 1);
            mma<type, a_read_per_thread, b_read_per_thread>(a_reg[j % 2], b_reg[j % 2], c_reg);
        }

        if (i < k)
        {
            store_next_a_data_to_smem<type, VEC_SIZE, a_read_per_thread, TILE_K, TILE_K * TILE_M>(next_data_a, a_tile[write_stage_idx], txa, tya);
            store_next_b_data_to_smem<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K * TILE_N>(next_data_b, b_tile[write_stage_idx], txb, tyb);
            __syncthreads();
            write_stage_idx ^= 1;
        }

        load_smem_to_a_reg<type, TILE_K * TILE_M, a_read_per_thread, TILE_M>(a_tile[load_stage_idx ^ 1], a_reg[0], tx, ty, 0);
        load_smem_to_b_reg<type, TILE_K * TILE_N, b_read_per_thread, TILE_N>(b_tile[load_stage_idx ^ 1], b_reg[0], tx, ty, 0);
        mma<type, a_read_per_thread, b_read_per_thread>(a_reg[1], b_reg[1], c_reg);
    } while (i < k);

    float *pC = C + blockIdx.y * TILE_M * n + blockIdx.x * TILE_N + tx * a_read_per_thread + ty * b_read_per_thread * n;

#pragma unroll
    for (int i = 0; i < a_read_per_thread; i++)
    {
#pragma unroll
        for (int j = 0; j < b_read_per_thread; j++)
        {
            pC[i * n + j] = c_reg[i][j];
        }
    }
}

#define BLOCK_X 16
#define BLOCK_Y 16

#define TILE_X 128
#define TILE_X_4 32
#define TILE_Y 128
#define TILE_Y_4 32

#define TILE_K 16

#define WPTN 8
#define WPTM 8
#define WPTN_4 2

extern "C" __global__ void gemm_kernel_NN(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float4 *__restrict__ C,
    float alpha, float beta,
    int M, int N, int K)
{
    __shared__ float4 smem_a[2][TILE_K * TILE_Y_4];
    __shared__ float4 smem_b[2][TILE_K * TILE_X_4];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;

    int tx4 = threadIdx.x % 4;
    int ty4 = threadIdx.x / 4;

    int tx32 = threadIdx.x % 32;
    int ty32 = threadIdx.x / 32;

    //! every thread block read TILE_Y rows of A
    //! every 4 thread read a row of A with TILE_K  elements
    //! every thread read 4 elements
    const float *pA = (A + K * TILE_Y * blockIdx.y + ty4 * K + tx4 * 4);
    //! every thread block read TILE_X columns of B
    //! every 32 thread read a row of B with TILE_X elements
    //! every thread read 4 elements
    const float *pB = (B + TILE_X * blockIdx.x + ty32 * N + tx32 * 4);

    //! every thread block write TILE_Y/4 rows of C, TILE_X_4 * 4(float4)
    //! columns of C
    float4 *pC = C + TILE_Y * blockIdx.y * N / 4 + TILE_X_4 * blockIdx.x;

    int sts_a_offset = tx4 * 4 * TILE_Y + ty4;
    int sts_b_offset = ty32 * TILE_X_4 + tx32;

    float4 f4_zero = make_float4(0.f, 0.f, 0.f, 0.f);
    bool valid_ld_a_0 = ((blockIdx.y * TILE_Y + ty4) < M) && ((tx4 * 4) < K);
    bool valid_ld_a_1 = ((blockIdx.y * TILE_Y + ty4 + 64) < M) && ((tx4 * 4) < K);
    bool valid_ld_b_0 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && (ty32 < K);
    bool valid_ld_b_1 = ((blockIdx.x * TILE_X + tx32 * 4) < N) && ((ty32 + 8) < K);

    float4 ldg_a_reg[2];
    float4 ldg_b_reg[2];

    ldg_a_reg[0] = valid_ld_a_0 ? *(const float4 *)pA : f4_zero;
    ldg_a_reg[1] = valid_ld_a_1 ? *(const float4 *)(pA + 64 * K) : f4_zero;
    ldg_b_reg[0] = valid_ld_b_0 ? *(const float4 *)(pB + 0 * N) : f4_zero;
    ldg_b_reg[1] = valid_ld_b_1 ? *(const float4 *)(pB + 8 * N) : f4_zero;

    float4 c[WPTM][WPTN_4] = {{f4_zero}};

    *((float *)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
    *((float *)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
    *((float *)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
    *((float *)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
    *((float *)&smem_a[0][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
    *((float *)&smem_a[0][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
    *((float *)&smem_a[0][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
    *((float *)&smem_a[0][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

    smem_b[0][sts_b_offset + 0] = ldg_b_reg[0];
    smem_b[0][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];

    __syncthreads();

    int i = 0;
    int write_stage_idx = 1;

    float4 reg_a[2][2];
    float4 reg_b[2][2];

    reg_a[0][0] = smem_a[0][0 + ty];
    reg_a[0][1] = smem_a[0][16 + ty];
    reg_b[0][0] = smem_b[0][0 + tx];
    reg_b[0][1] = smem_b[0][16 + tx];

    do
    {
        i += 16;
        valid_ld_a_0 = (valid_ld_a_0 && ((tx4 * 4 + i) < K));
        valid_ld_a_1 = (valid_ld_a_1 && ((tx4 * 4 + i) < K));
        valid_ld_b_0 = (valid_ld_b_0 && ((ty32 + i) < K));
        valid_ld_b_1 = (valid_ld_b_1 && ((ty32 + 8 + i) < K));

        ldg_a_reg[0] = (valid_ld_a_0) ? *(const float4 *)(pA + i + 0) : f4_zero;
        ldg_a_reg[1] = (valid_ld_a_1) ? *(const float4 *)(pA + i + 64 * K) : f4_zero;
        ldg_b_reg[0] = (valid_ld_b_0) ? *(const float4 *)(pB + (i + 0) * N) : f4_zero;
        ldg_b_reg[1] = (valid_ld_b_1) ? *(const float4 *)(pB + (i + 8) * N) : f4_zero;

        int load_stage_idx = write_stage_idx ^ 1;

#pragma unroll
        for (int j = 0; j < TILE_K - 1; j++)
        {
            reg_a[(j + 1) % 2][0] = smem_a[load_stage_idx][(j + 1) * TILE_Y_4 + 0 + ty];
            reg_a[(j + 1) % 2][1] = smem_a[load_stage_idx][(j + 1) * TILE_Y_4 + 16 + ty];
            reg_b[(j + 1) % 2][0] = smem_b[load_stage_idx][(j + 1) * TILE_X_4 + 0 + tx];
            reg_b[(j + 1) % 2][1] = smem_b[load_stage_idx][(j + 1) * TILE_X_4 + 16 + tx];
            c[0][0].x += reg_a[j % 2][0].x * reg_b[j % 2][0].x;
            c[0][0].y += reg_a[j % 2][0].x * reg_b[j % 2][0].y;
            c[0][0].z += reg_a[j % 2][0].x * reg_b[j % 2][0].z;
            c[0][0].w += reg_a[j % 2][0].x * reg_b[j % 2][0].w;
            c[0][1].x += reg_a[j % 2][0].x * reg_b[j % 2][1].x;
            c[0][1].y += reg_a[j % 2][0].x * reg_b[j % 2][1].y;
            c[0][1].z += reg_a[j % 2][0].x * reg_b[j % 2][1].z;
            c[0][1].w += reg_a[j % 2][0].x * reg_b[j % 2][1].w;
            c[1][0].x += reg_a[j % 2][0].y * reg_b[j % 2][0].x;
            c[1][0].y += reg_a[j % 2][0].y * reg_b[j % 2][0].y;
            c[1][0].z += reg_a[j % 2][0].y * reg_b[j % 2][0].z;
            c[1][0].w += reg_a[j % 2][0].y * reg_b[j % 2][0].w;
            c[1][1].x += reg_a[j % 2][0].y * reg_b[j % 2][1].x;
            c[1][1].y += reg_a[j % 2][0].y * reg_b[j % 2][1].y;
            c[1][1].z += reg_a[j % 2][0].y * reg_b[j % 2][1].z;
            c[1][1].w += reg_a[j % 2][0].y * reg_b[j % 2][1].w;
            c[2][0].x += reg_a[j % 2][0].z * reg_b[j % 2][0].x;
            c[2][0].y += reg_a[j % 2][0].z * reg_b[j % 2][0].y;
            c[2][0].z += reg_a[j % 2][0].z * reg_b[j % 2][0].z;
            c[2][0].w += reg_a[j % 2][0].z * reg_b[j % 2][0].w;
            c[2][1].x += reg_a[j % 2][0].z * reg_b[j % 2][1].x;
            c[2][1].y += reg_a[j % 2][0].z * reg_b[j % 2][1].y;
            c[2][1].z += reg_a[j % 2][0].z * reg_b[j % 2][1].z;
            c[2][1].w += reg_a[j % 2][0].z * reg_b[j % 2][1].w;
            c[3][0].x += reg_a[j % 2][0].w * reg_b[j % 2][0].x;
            c[3][0].y += reg_a[j % 2][0].w * reg_b[j % 2][0].y;
            c[3][0].z += reg_a[j % 2][0].w * reg_b[j % 2][0].z;
            c[3][0].w += reg_a[j % 2][0].w * reg_b[j % 2][0].w;
            c[3][1].x += reg_a[j % 2][0].w * reg_b[j % 2][1].x;
            c[3][1].y += reg_a[j % 2][0].w * reg_b[j % 2][1].y;
            c[3][1].z += reg_a[j % 2][0].w * reg_b[j % 2][1].z;
            c[3][1].w += reg_a[j % 2][0].w * reg_b[j % 2][1].w;
            c[4][0].x += reg_a[j % 2][1].x * reg_b[j % 2][0].x;
            c[4][0].y += reg_a[j % 2][1].x * reg_b[j % 2][0].y;
            c[4][0].z += reg_a[j % 2][1].x * reg_b[j % 2][0].z;
            c[4][0].w += reg_a[j % 2][1].x * reg_b[j % 2][0].w;
            c[4][1].x += reg_a[j % 2][1].x * reg_b[j % 2][1].x;
            c[4][1].y += reg_a[j % 2][1].x * reg_b[j % 2][1].y;
            c[4][1].z += reg_a[j % 2][1].x * reg_b[j % 2][1].z;
            c[4][1].w += reg_a[j % 2][1].x * reg_b[j % 2][1].w;
            c[5][0].x += reg_a[j % 2][1].y * reg_b[j % 2][0].x;
            c[5][0].y += reg_a[j % 2][1].y * reg_b[j % 2][0].y;
            c[5][0].z += reg_a[j % 2][1].y * reg_b[j % 2][0].z;
            c[5][0].w += reg_a[j % 2][1].y * reg_b[j % 2][0].w;
            c[5][1].x += reg_a[j % 2][1].y * reg_b[j % 2][1].x;
            c[5][1].y += reg_a[j % 2][1].y * reg_b[j % 2][1].y;
            c[5][1].z += reg_a[j % 2][1].y * reg_b[j % 2][1].z;
            c[5][1].w += reg_a[j % 2][1].y * reg_b[j % 2][1].w;
            c[6][0].x += reg_a[j % 2][1].z * reg_b[j % 2][0].x;
            c[6][0].y += reg_a[j % 2][1].z * reg_b[j % 2][0].y;
            c[6][0].z += reg_a[j % 2][1].z * reg_b[j % 2][0].z;
            c[6][0].w += reg_a[j % 2][1].z * reg_b[j % 2][0].w;
            c[6][1].x += reg_a[j % 2][1].z * reg_b[j % 2][1].x;
            c[6][1].y += reg_a[j % 2][1].z * reg_b[j % 2][1].y;
            c[6][1].z += reg_a[j % 2][1].z * reg_b[j % 2][1].z;
            c[6][1].w += reg_a[j % 2][1].z * reg_b[j % 2][1].w;
            c[7][0].x += reg_a[j % 2][1].w * reg_b[j % 2][0].x;
            c[7][0].y += reg_a[j % 2][1].w * reg_b[j % 2][0].y;
            c[7][0].z += reg_a[j % 2][1].w * reg_b[j % 2][0].z;
            c[7][0].w += reg_a[j % 2][1].w * reg_b[j % 2][0].w;
            c[7][1].x += reg_a[j % 2][1].w * reg_b[j % 2][1].x;
            c[7][1].y += reg_a[j % 2][1].w * reg_b[j % 2][1].y;
            c[7][1].z += reg_a[j % 2][1].w * reg_b[j % 2][1].z;
            c[7][1].w += reg_a[j % 2][1].w * reg_b[j % 2][1].w;
        }

        //! the last iter K, write the global data to shared memory which will
        //! be used in the next iteration
        if (i < K)
        {
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 0) = ldg_a_reg[0].x;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 0) = ldg_a_reg[0].y;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 0) = ldg_a_reg[0].z;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 0) = ldg_a_reg[0].w;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 0 * TILE_Y + 64) = ldg_a_reg[1].x;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 1 * TILE_Y + 64) = ldg_a_reg[1].y;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 2 * TILE_Y + 64) = ldg_a_reg[1].z;
            *((float *)&smem_a[write_stage_idx][0] + sts_a_offset + 3 * TILE_Y + 64) = ldg_a_reg[1].w;

            smem_b[write_stage_idx][sts_b_offset + 0] = ldg_b_reg[0];
            smem_b[write_stage_idx][sts_b_offset + 8 * TILE_X_4] = ldg_b_reg[1];
            __syncthreads();
            write_stage_idx ^= 1;
        }

        //! load data from shared memory to register for the next TILE_K
        //! iteration
        reg_a[0][0] = smem_a[load_stage_idx ^ 1][0 + ty];
        reg_a[0][1] = smem_a[load_stage_idx ^ 1][16 + ty];
        reg_b[0][0] = smem_b[load_stage_idx ^ 1][0 + tx];
        reg_b[0][1] = smem_b[load_stage_idx ^ 1][16 + tx];

        //! compute the last TILE_K-1 iteration, the register data is load ahead
        c[0][0].x += reg_a[1][0].x * reg_b[1][0].x;
        c[0][0].y += reg_a[1][0].x * reg_b[1][0].y;
        c[0][0].z += reg_a[1][0].x * reg_b[1][0].z;
        c[0][0].w += reg_a[1][0].x * reg_b[1][0].w;
        c[0][1].x += reg_a[1][0].x * reg_b[1][1].x;
        c[0][1].y += reg_a[1][0].x * reg_b[1][1].y;
        c[0][1].z += reg_a[1][0].x * reg_b[1][1].z;
        c[0][1].w += reg_a[1][0].x * reg_b[1][1].w;
        c[1][0].x += reg_a[1][0].y * reg_b[1][0].x;
        c[1][0].y += reg_a[1][0].y * reg_b[1][0].y;
        c[1][0].z += reg_a[1][0].y * reg_b[1][0].z;
        c[1][0].w += reg_a[1][0].y * reg_b[1][0].w;
        c[1][1].x += reg_a[1][0].y * reg_b[1][1].x;
        c[1][1].y += reg_a[1][0].y * reg_b[1][1].y;
        c[1][1].z += reg_a[1][0].y * reg_b[1][1].z;
        c[1][1].w += reg_a[1][0].y * reg_b[1][1].w;
        c[2][0].x += reg_a[1][0].z * reg_b[1][0].x;
        c[2][0].y += reg_a[1][0].z * reg_b[1][0].y;
        c[2][0].z += reg_a[1][0].z * reg_b[1][0].z;
        c[2][0].w += reg_a[1][0].z * reg_b[1][0].w;
        c[2][1].x += reg_a[1][0].z * reg_b[1][1].x;
        c[2][1].y += reg_a[1][0].z * reg_b[1][1].y;
        c[2][1].z += reg_a[1][0].z * reg_b[1][1].z;
        c[2][1].w += reg_a[1][0].z * reg_b[1][1].w;
        c[3][0].x += reg_a[1][0].w * reg_b[1][0].x;
        c[3][0].y += reg_a[1][0].w * reg_b[1][0].y;
        c[3][0].z += reg_a[1][0].w * reg_b[1][0].z;
        c[3][0].w += reg_a[1][0].w * reg_b[1][0].w;
        c[3][1].x += reg_a[1][0].w * reg_b[1][1].x;
        c[3][1].y += reg_a[1][0].w * reg_b[1][1].y;
        c[3][1].z += reg_a[1][0].w * reg_b[1][1].z;
        c[3][1].w += reg_a[1][0].w * reg_b[1][1].w;
        c[4][0].x += reg_a[1][1].x * reg_b[1][0].x;
        c[4][0].y += reg_a[1][1].x * reg_b[1][0].y;
        c[4][0].z += reg_a[1][1].x * reg_b[1][0].z;
        c[4][0].w += reg_a[1][1].x * reg_b[1][0].w;
        c[4][1].x += reg_a[1][1].x * reg_b[1][1].x;
        c[4][1].y += reg_a[1][1].x * reg_b[1][1].y;
        c[4][1].z += reg_a[1][1].x * reg_b[1][1].z;
        c[4][1].w += reg_a[1][1].x * reg_b[1][1].w;
        c[5][0].x += reg_a[1][1].y * reg_b[1][0].x;
        c[5][0].y += reg_a[1][1].y * reg_b[1][0].y;
        c[5][0].z += reg_a[1][1].y * reg_b[1][0].z;
        c[5][0].w += reg_a[1][1].y * reg_b[1][0].w;
        c[5][1].x += reg_a[1][1].y * reg_b[1][1].x;
        c[5][1].y += reg_a[1][1].y * reg_b[1][1].y;
        c[5][1].z += reg_a[1][1].y * reg_b[1][1].z;
        c[5][1].w += reg_a[1][1].y * reg_b[1][1].w;
        c[6][0].x += reg_a[1][1].z * reg_b[1][0].x;
        c[6][0].y += reg_a[1][1].z * reg_b[1][0].y;
        c[6][0].z += reg_a[1][1].z * reg_b[1][0].z;
        c[6][0].w += reg_a[1][1].z * reg_b[1][0].w;
        c[6][1].x += reg_a[1][1].z * reg_b[1][1].x;
        c[6][1].y += reg_a[1][1].z * reg_b[1][1].y;
        c[6][1].z += reg_a[1][1].z * reg_b[1][1].z;
        c[6][1].w += reg_a[1][1].z * reg_b[1][1].w;
        c[7][0].x += reg_a[1][1].w * reg_b[1][0].x;
        c[7][0].y += reg_a[1][1].w * reg_b[1][0].y;
        c[7][0].z += reg_a[1][1].w * reg_b[1][0].z;
        c[7][0].w += reg_a[1][1].w * reg_b[1][0].w;
        c[7][1].x += reg_a[1][1].w * reg_b[1][1].x;
        c[7][1].y += reg_a[1][1].w * reg_b[1][1].y;
        c[7][1].z += reg_a[1][1].w * reg_b[1][1].z;
        c[7][1].w += reg_a[1][1].w * reg_b[1][1].w;

    } while (i < K);

#pragma unroll
    for (int wm = 0; wm < WPTM; wm++)
    {
#pragma unroll
        for (int wn = 0; wn < WPTN_4; wn++)
        {
            c[wm][wn].x *= alpha;
            c[wm][wn].y *= alpha;
            c[wm][wn].z *= alpha;
            c[wm][wn].w *= alpha;
        }
    }

#pragma unroll
    for (int wm = 0; wm < 4; wm++)
    {
#pragma unroll
        for (int wn = 0; wn < WPTN_4; wn++)
        {
            if (((blockIdx.y * TILE_Y + ty * 4 + wm) < M) && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N))
            {
                if (beta != 0)
                {
                    float4 vec4c = *(pC + ((ty * 4 + wm) * N / 4 + wn * 16 + tx));
                    vec4c.x = vec4c.x * beta + c[wm][wn].x;
                    vec4c.y = vec4c.y * beta + c[wm][wn].y;
                    vec4c.z = vec4c.z * beta + c[wm][wn].z;
                    vec4c.w = vec4c.w * beta + c[wm][wn].w;
                    *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                }
                else
                {
                    *(pC + (ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm][wn];
                }
            }
        }
    }

#pragma unroll
    for (int wm = 0; wm < 4; wm++)
    {
#pragma unroll
        for (int wn = 0; wn < WPTN_4; wn++)
        {
            if (((blockIdx.y * TILE_Y + 64 + ty * 4 + wm) < M) && ((blockIdx.x * TILE_X + wn * 64 + tx * 4) < N))
            {
                if (beta != 0)
                {
                    float4 vec4c = *(pC + ((64 + ty * 4 + wm) * N / 4 + wn * 16 + tx));
                    vec4c.x = vec4c.x * beta + c[wm + 4][wn].x;
                    vec4c.y = vec4c.y * beta + c[wm + 4][wn].y;
                    vec4c.z = vec4c.z * beta + c[wm + 4][wn].z;
                    vec4c.w = vec4c.w * beta + c[wm + 4][wn].w;
                    *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = vec4c;
                }
                else
                {
                    *(pC + (64 + ty * 4 + wm) * N / 4 + wn * 16 + tx) = c[wm + 4][wn];
                }
            }
        }
    }
}