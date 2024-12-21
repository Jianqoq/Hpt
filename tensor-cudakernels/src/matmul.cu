#define type float

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

// 定义 Tile 和线程处理大小
#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define THREAD_TILE_SIZE 16 // 每个线程计算16x16的输出块

// 矩阵乘法内核
extern "C" __global__ void matmul_blocked2(
    const type *A, const type *B, type *C,
    size_t m, size_t n, size_t k, size_t batch_size)
{
    // 定义共享内存
    __shared__ type a_tile[TILE_M][TILE_K];
    __shared__ type b_tile[TILE_K][TILE_N];

    // 检查 batch 索引
    if (blockIdx.z >= batch_size)
        return;

    // 计算每个线程负责的起始行和列
    size_t row_base = (blockIdx.y * blockDim.y + threadIdx.y) * THREAD_TILE_SIZE;
    size_t col_base = (blockIdx.x * blockDim.x + threadIdx.x) * THREAD_TILE_SIZE;

    if (row_base >= m || col_base >= n)
        return;

    // 初始化 4x4 的累加器
    type c_reg[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {{0.0f}};

    // 计算需要的 Tile 数
    size_t num_tiles = (k + TILE_K - 1) / TILE_K;

    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++)
    {
        size_t j = tile_idx * TILE_K;

#pragma unroll
        for (int i = 0; i < THREAD_TILE_SIZE; i++)
        {
            size_t a_row = row_base + i;
#pragma unroll
            for (size_t b = 0; b < TILE_K; b++)
            {
                size_t a_col = j + b;
                if (a_row < m && a_col < k)
                {
                    a_tile[threadIdx.y * THREAD_TILE_SIZE + i][b] = A[blockIdx.z * m * k + a_row * k + a_col];
                }
                else
                {
                    a_tile[threadIdx.y * THREAD_TILE_SIZE + i][b] = 0.0f;
                }
            }
        }

#pragma unroll
        for (int i = 0; i < THREAD_TILE_SIZE; i++)
        {
            size_t b_col = col_base + i;
#pragma unroll
            for (size_t b = 0; b < TILE_K; b++)
            {
                size_t b_row = j + b;
                if (b_row < k && b_col < n)
                {
                    b_tile[b][threadIdx.x * THREAD_TILE_SIZE + i] = B[blockIdx.z * k * n + b_row * n + b_col];
                }
                else
                {
                    b_tile[b][threadIdx.x * THREAD_TILE_SIZE + i] = 0.0f;
                }
            }
        }
        __syncthreads(); // 确保所有数据都加载完成

#pragma unroll
        for (int t = 0; t < TILE_K; t++)
        {
#pragma unroll
            for (int i = 0; i < THREAD_TILE_SIZE; i++)
            {
#pragma unroll
                for (int j_inner = 0; j_inner < THREAD_TILE_SIZE; j_inner++)
                {
                    c_reg[i][j_inner] += a_tile[threadIdx.y * THREAD_TILE_SIZE + i][t] * b_tile[t][threadIdx.x * THREAD_TILE_SIZE + j_inner];
                }
            }
        }

        __syncthreads(); // 确保所有计算完成
    }
#pragma unroll
    for (int i = 0; i < THREAD_TILE_SIZE; i++)
    {
#pragma unroll
        for (int j_inner = 0; j_inner < THREAD_TILE_SIZE; j_inner++)
        {
            size_t current_row = row_base + i;
            size_t current_col = col_base + j_inner;
            if (current_row < m && current_col < n)
            {
                C[blockIdx.z * m * n + current_row * n + current_col] = c_reg[i][j_inner];
            }
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