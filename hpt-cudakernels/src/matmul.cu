// #include <stdio.h>
// #include <assert.h>
// #define type float
// #define type_vec float4

// extern "C" __global__ void matmul_naive(type *a, type *b, type *out, size_t m, size_t n, size_t k, size_t batch_size)
// {
//     if (blockIdx.z >= batch_size)
//         return;

//     size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//     size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row >= m || col >= n)
//         return;

//     type value = 0.0f;
//     for (size_t i = 0; i < k; i++)
//     {
//         value += a[blockIdx.z * m * k + row * k + i] * b[blockIdx.z * k * n + i * n + col];
//     }

//     out[blockIdx.z * m * n + row * n + col] = value;
// }

// #define TILE_SIZE 16

// __device__ void mm4x4(type a[4], type b[4], type out[4][4])
// {
// #pragma unroll
//     for (size_t i = 0; i < 4; i++)
//     {
//         out[i][0] += a[i] * b[0];
//         out[i][1] += a[i] * b[1];
//         out[i][2] += a[i] * b[2];
//         out[i][3] += a[i] * b[3];
//     }
// }

// extern "C" __global__ void matmul_blocked(type *a, type *b, type *out, size_t m, size_t n, size_t k, size_t batch_size)
// {

//     __shared__ type a_tile[TILE_SIZE][TILE_SIZE];
//     __shared__ type b_tile[TILE_SIZE][TILE_SIZE];

//     if (blockIdx.z >= batch_size)
//         return;

//     size_t row = blockIdx.y * blockDim.y + threadIdx.y;
//     size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row >= m || col >= n)
//         return;

//     type value = 0.0f;
//     size_t end = (k + TILE_SIZE - 1) / TILE_SIZE;
//     for (size_t i = 0; i < end; i++)
//     {
//         if (i * TILE_SIZE + threadIdx.x < k)
//         {
//             a_tile[threadIdx.y][threadIdx.x] = a[blockIdx.z * m * k + row * k + i * TILE_SIZE + threadIdx.x];
//         }
//         else
//         {
//             a_tile[threadIdx.y][threadIdx.x] = 0.0f;
//         }
//         if (i * TILE_SIZE + threadIdx.y < k)
//         {
//             b_tile[threadIdx.y][threadIdx.x] = b[blockIdx.z * k * n + (i * TILE_SIZE + threadIdx.y) * n + col];
//         }
//         else
//         {
//             b_tile[threadIdx.y][threadIdx.x] = 0.0f;
//         }

//         __syncthreads();

// #pragma unroll
//         for (size_t j = 0; j < TILE_SIZE; j++)
//         {
//             value += a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x];
//         }
//         __syncthreads();
//     }

//     out[blockIdx.z * m * n + row * n + col] = value;
// }

// template <size_t M, size_t N, size_t K, size_t TileSize>
// __device__ void load_to_smem(
//     const type *A, const type *B,
//     size_t j, size_t m, size_t n, size_t k,
//     size_t row, size_t col,
//     type a_tile[M][K], type b_tile[K][N])
// {
// // 加载 A 块到共享内存
// #pragma unroll
//     for (int i = 0; i < TileSize; i++)
//     {
//         size_t a_row = row + threadIdx.y * TileSize + i;
//         size_t a_col = j + threadIdx.x;
//         if (a_row < m && a_col < k)
//         {
//             a_tile[threadIdx.y * TileSize + i][threadIdx.x] = A[blockIdx.z * m * k + a_row * k + a_col];
//         }
//         else
//         {
//             a_tile[threadIdx.y * TileSize + i][threadIdx.x] = 0.0f;
//         }
//     }

// // 加载 B 块到共享内存
// #pragma unroll
//     for (int i = 0; i < TileSize; i++)
//     {
//         size_t b_row = j + threadIdx.y * TileSize + i;
//         size_t b_col = col + threadIdx.x;
//         if (b_row < k && b_col < n)
//         {
//             b_tile[threadIdx.y * TileSize + i][threadIdx.x] = B[blockIdx.z * k * n + b_row * n + b_col];
//         }
//         else
//         {
//             b_tile[threadIdx.y * TileSize + i][threadIdx.x] = 0.0f;
//         }
//     }
// }

// // define tile and thread processing size
// #define TILE_M 128
// #define TILE_N 128
// #define TILE_K 16
// #define THREAD_BLOCK_X 16
// #define THREAD_BLOCK_Y 16
// #define VEC_SIZE 4

// template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileM, size_t TileK>
// __device__ void load_gmem_to_next_a_reg(const T *A, T next_data_a[ReadPerThread], size_t txa, size_t tya, size_t k, const size_t tile_idx)
// {
//     const float *pA = (A + (txa * ReadPerThread + tile_idx) + (blockIdx.y * TileM + tya) * k);
// #pragma unroll
//     for (int i = 0; i < ReadPerThread; i++)
//     {
//         next_data_a[i] = pA[i];
//     }
// }

// template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileN, size_t TileK>
// __device__ void load_gmem_to_next_b_reg(const T *B, T next_data_b[ReadPerThread], size_t txb, size_t tyb, size_t n, const size_t tile_idx)
// {
//     const float *pB = (B + txb * ReadPerThread + blockIdx.x * TileN + tyb * n + tile_idx * n);
// #pragma unroll
//     for (int i = 0; i < ReadPerThread; i++)
//     {
//         next_data_b[i] = pB[i];
//     }
// }

// template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileM, size_t smem_size>
// __device__ void store_next_a_data_to_smem(T next_data[ReadPerThread], T smem[smem_size], size_t txa, size_t tya)
// {
// #pragma unroll
//     for (int i = 0; i < ReadPerThread; i++)
//     {
//         smem[(txa * ReadPerThread + i) * TileM + tya] = next_data[i];
//     }
// }

// template <typename T, size_t VecSize, size_t ReadPerThread, size_t TileN, size_t smem_size>
// __device__ void store_next_b_data_to_smem(T next_data[ReadPerThread], T smem[smem_size], size_t txb, size_t tyb)
// {
// #pragma unroll
//     for (int i = 0; i < ReadPerThread; i++)
//     {
//         smem[txb * ReadPerThread + i + tyb * TileN] = next_data[i];
//     }
// }

// template <typename T, size_t smem_size, size_t ReadPerThread, size_t TileM>
// __device__ void load_smem_to_a_reg(T smem[smem_size], T reg[ReadPerThread], size_t tx, size_t ty, size_t j)
// {
// #pragma unroll
//     for (int i = 0; i < ReadPerThread; i++)
//     {
//         reg[i] = smem[ty * ReadPerThread + i + TileM * j];
//     }
// }

// template <typename T, size_t smem_size, size_t ReadPerThread, size_t TileN>
// __device__ void load_smem_to_b_reg(T smem[smem_size], T reg[ReadPerThread], size_t tx, size_t ty, size_t j)
// {
// #pragma unroll
//     for (int i = 0; i < ReadPerThread; i++)
//     {
//         reg[i] = smem[tx * ReadPerThread + i + TileN * j];
//     }
// }

// template <typename T, size_t AReadPerThread, size_t BReadPerThread>
// __device__ void mma(T a[AReadPerThread], T b[BReadPerThread], T c[AReadPerThread][BReadPerThread])
// {
// #pragma unroll
//     for (size_t i = 0; i < AReadPerThread; ++i)
//     {
// #pragma unroll
//         for (size_t j = 0; j < BReadPerThread; ++j)
//         {
//             c[i][j] += a[i] * b[j];
//         }
//     }
// }

// __device__ int lock = 0;
// // 获取锁
// __device__ void acquire_lock(int *lock)
// {
//     while (atomicCAS(lock, 0, 1) != 0)
//     {
//         // 自旋等待锁释放
//     }
// }

// // 释放锁
// __device__ void release_lock(int *lock)
// {
//     atomicExch(lock, 0);
// }

// // matrix multiplication kernel
// extern "C" __global__ void matmul_blocked2(
//     const type *__restrict__ A, const type *__restrict__ B, type *__restrict__ C,
//     size_t m, size_t n, size_t k, size_t batch_size)
// {
//     // define shared memory
//     __shared__ type a_tile[2][TILE_K * TILE_M];
//     __shared__ type b_tile[2][TILE_K * TILE_N];

//     constexpr const size_t a_read_per_thread = TILE_M * TILE_K / (THREAD_BLOCK_X * THREAD_BLOCK_Y);
//     constexpr const size_t b_read_per_thread = TILE_N * TILE_K / (THREAD_BLOCK_X * THREAD_BLOCK_Y);

//     static_assert(TILE_M % a_read_per_thread == 0, "a_read_per_thread must be multiple of TILE_M");
//     static_assert(TILE_K % b_read_per_thread == 0, "b_read_per_thread must be multiple of TILE_K");
//     static_assert(a_read_per_thread < TILE_K, "a_read_per_thread must be less than TILE_K");
//     static_assert(b_read_per_thread < TILE_K, "b_read_per_thread must be less than TILE_K");

//     type next_data_a[a_read_per_thread];
//     type next_data_b[b_read_per_thread];
//     type a_reg[2][a_read_per_thread]; // must be multiple of VEC_SIZE
//     type b_reg[2][b_read_per_thread]; // must be multiple of VEC_SIZE

//     // // initialize 4x4 accumulator
//     type c_reg[a_read_per_thread][b_read_per_thread] = {{0.0f}};

//     size_t txa = threadIdx.x % (TILE_K / a_read_per_thread);
//     size_t tya = threadIdx.x / (TILE_K / a_read_per_thread);

//     size_t txb = threadIdx.x % (TILE_N / b_read_per_thread);
//     size_t tyb = threadIdx.x / (TILE_N / b_read_per_thread);
//     // load data from global memory to next_data_regs, each next_data_a store TILE_K elements

//     load_gmem_to_next_a_reg<type, VEC_SIZE, a_read_per_thread, TILE_M, TILE_K>(A, next_data_a, txa, tya, k, 0);
//     load_gmem_to_next_b_reg<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K>(B, next_data_b, txb, tyb, n, 0);

//     // int global_idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * (gridDim.x * blockDim.x);

//     store_next_a_data_to_smem<type, VEC_SIZE, a_read_per_thread, TILE_M, TILE_K * TILE_M>(next_data_a, a_tile[0], txa, tya);
//     store_next_b_data_to_smem<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K * TILE_N>(next_data_b, b_tile[0], txb, tyb);
//     size_t tx = threadIdx.x % THREAD_BLOCK_X;
//     size_t ty = threadIdx.x / THREAD_BLOCK_X;

//     __syncthreads();

//     load_smem_to_a_reg<type, TILE_K * TILE_M, a_read_per_thread, TILE_M>(a_tile[0], a_reg[0], tx, ty, 0);
//     load_smem_to_b_reg<type, TILE_K * TILE_N, b_read_per_thread, TILE_N>(b_tile[0], b_reg[0], tx, ty, 0);

//     // acquire_lock(&lock);
//     // if ((threadIdx.x == 0) && blockIdx.x == 1 && blockIdx.y == 0)
//     // {
//     //     printf("global_idx = %d, A: [", global_idx);
//     //     for (int i = 0; i < a_read_per_thread; i++)
//     //     {
//     //         printf("%f, ", next_data_a[i]);
//     //     }
//     //     printf("], B: [");
//     //     for (int i = 0; i < b_read_per_thread; i++)
//     //     {
//     //         printf("%f, ", next_data_b[i]);
//     //     }
//     //     printf("], (%llu, %llu)\n", tx, ty);
//     //     // printf("a_tile[0]:\n");
//     //     // for (int i = 0; i < TILE_K; i++)
//     //     // {
//     //     //     for (int j = 0; j < TILE_M; j++)
//     //     //     {
//     //     //         printf("%f, ", a_tile[0][i * TILE_M + j]);
//     //     //     }
//     //     //     printf("\n");
//     //     // }
//     //     // printf("\n");
//     //     // printf("b_tile[0]:\n");
//     //     // for (int i = 0; i < TILE_K; i++)
//     //     // {
//     //     //     for (int j = 0; j < TILE_N; j++)
//     //     //     {
//     //     //         printf("%f, ", b_tile[0][i * TILE_N + j]);
//     //     //     }
//     //     //     printf("\n");
//     //     // }
//     //     // printf("\n");
//     //     printf("a_reg[0]:\n");
//     //     for (int i = 0; i < a_read_per_thread; i++)
//     //     {
//     //         printf("%f, ", a_reg[0][i]);
//     //     }
//     //     printf("\n");
//     //     // printf("b_reg[0]:\n");
//     //     // for (int i = 0; i < b_read_per_thread; i++)
//     //     // {
//     //     //     printf("%f, ", b_reg[0][i]);
//     //     // }
//     //     // printf("\n");
//     // }
//     // release_lock(&lock);

//     int i = 0;
//     int write_stage_idx = 1;
//     do
//     {
//         i += TILE_K;

//         load_gmem_to_next_a_reg<type, VEC_SIZE, a_read_per_thread, TILE_M, TILE_K>(A, next_data_a, txa, tya, k, i);
//         load_gmem_to_next_b_reg<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K>(B, next_data_b, txb, tyb, n, i);

//         int load_stage_idx = write_stage_idx ^ 1;

//         for (int j = 0; j < TILE_K - 1; j++)
//         {
//             load_smem_to_a_reg<type, TILE_K * TILE_M, a_read_per_thread, TILE_M>(a_tile[load_stage_idx], a_reg[(j + 1) % 2], tx, ty, j + 1);
//             load_smem_to_b_reg<type, TILE_K * TILE_N, b_read_per_thread, TILE_N>(b_tile[load_stage_idx], b_reg[(j + 1) % 2], tx, ty, j + 1);
//             mma<type, a_read_per_thread, b_read_per_thread>(a_reg[j % 2], b_reg[j % 2], c_reg);
//         }

//         if (i < k)
//         {
//             store_next_a_data_to_smem<type, VEC_SIZE, a_read_per_thread, TILE_M, TILE_K * TILE_M>(next_data_a, a_tile[write_stage_idx], txa, tya);
//             store_next_b_data_to_smem<type, VEC_SIZE, b_read_per_thread, TILE_N, TILE_K * TILE_N>(next_data_b, b_tile[write_stage_idx], txb, tyb);
//             __syncthreads();
//             write_stage_idx ^= 1;
//         }

//         load_smem_to_a_reg<type, TILE_K * TILE_M, a_read_per_thread, TILE_M>(a_tile[load_stage_idx ^ 1], a_reg[0], tx, ty, 0);
//         load_smem_to_b_reg<type, TILE_K * TILE_N, b_read_per_thread, TILE_N>(b_tile[load_stage_idx ^ 1], b_reg[0], tx, ty, 0);
//         mma<type, a_read_per_thread, b_read_per_thread>(a_reg[1], b_reg[1], c_reg);
//     } while (i < k);

//     float *pC = C + blockIdx.y * TILE_M * n + blockIdx.x * TILE_N + tx * a_read_per_thread + ty * b_read_per_thread * n;

// #pragma unroll
//     for (int i = 0; i < a_read_per_thread; i++)
//     {
// #pragma unroll
//         for (int j = 0; j < b_read_per_thread; j++)
//         {
//             pC[i * n + j] = c_reg[i][j];
//         }
//     }
// }

// #define LOAD_GMEM_TO_NEXT_A_REG_VEC(VecSize, ReadPerThread, TileM, TileK, A, next_data_a, txa, tya, k, tile_idx)       \
//     {                                                                                                                  \
//         const float *pA = (A + (txa * ReadPerThread + tile_idx) + (blockIdx.y * TileM + tya) * k);                     \
//         _Pragma("unroll") for (int j = 0; j < ReadPerThread / VecSize; j++)                                            \
//         {                                                                                                              \
//             /*assert(((txa * ReadPerThread + tile_idx) + (blockIdx.y * TileM + tya) * k + j * VecSize + 3) < m * k);*/ \
//             next_data_a[j].x = pA[j * VecSize + 0];                                                                    \
//             next_data_a[j].y = pA[j * VecSize + 1];                                                                    \
//             next_data_a[j].z = pA[j * VecSize + 2];                                                                    \
//             next_data_a[j].w = pA[j * VecSize + 3];                                                                    \
//         }                                                                                                              \
//     }

// #define LOAD_GMEM_TO_NEXT_B_REG_VEC(VecSize, ReadPerThread, TileN, TileK, B, next_data_b, txb, tyb, n, tile_idx)       \
//     {                                                                                                                  \
//         const float *pB = (B + txb * ReadPerThread + blockIdx.x * TileN + tyb * n + tile_idx * n);                     \
//         _Pragma("unroll") for (int j = 0; j < ReadPerThread / VecSize; j++)                                            \
//         {                                                                                                              \
//             /*assert((txb * ReadPerThread + blockIdx.x * TileN + tyb * n + tile_idx * n + j * VecSize + 3) < n * k);*/ \
//             next_data_b[j].x = pB[j * VecSize + 0];                                                                    \
//             next_data_b[j].y = pB[j * VecSize + 1];                                                                    \
//             next_data_b[j].z = pB[j * VecSize + 2];                                                                    \
//             next_data_b[j].w = pB[j * VecSize + 3];                                                                    \
//         }                                                                                                              \
//     }

// #define STORE_NEXT_A_DATA_TO_SMEM_VEC(T, VecSize, ReadPerThread, TileM, next_data, smem, txa, tya)  \
//     {                                                                                               \
//         _Pragma("unroll") for (int i = 0; i < ReadPerThread / VecSize; i++)                         \
//         {                                                                                           \
//             /*assert(((txa * ReadPerThread + i * VecSize + 3) * TILE_M + tya) < TILE_K * TILE_M);*/ \
//             ((T *)&smem)[(txa * ReadPerThread + i * VecSize + 0) * TileM + tya] = next_data[i].x;   \
//             ((T *)&smem)[(txa * ReadPerThread + i * VecSize + 1) * TileM + tya] = next_data[i].y;   \
//             ((T *)&smem)[(txa * ReadPerThread + i * VecSize + 2) * TileM + tya] = next_data[i].z;   \
//             ((T *)&smem)[(txa * ReadPerThread + i * VecSize + 3) * TileM + tya] = next_data[i].w;   \
//         }                                                                                           \
//     }

// #define STORE_NEXT_B_DATA_TO_SMEM_VEC(VecSize, ReadPerThread, TileN, next_data, smem, txb, tyb)                          \
//     {                                                                                                                    \
//         _Pragma("unroll") for (int i = 0; i < ReadPerThread / VecSize; i++)                                              \
//         {                                                                                                                \
//             /*assert((txb * ReadPerThread + i * VecSize + tyb * (TileN / ReadPerThread)) < TILE_K * TILE_N / VecSize);*/ \
//             smem[txb * ReadPerThread / VecSize + i + tyb * (TileN / VecSize)] = next_data[i];                            \
//         }                                                                                                                \
//     }

// #define LOAD_SMEM_TO_A_REG_VEC(VecSize, ReadPerThread, TileM, reg, smem, ty, j)                                     \
//     {                                                                                                               \
//         _Pragma("unroll") for (int i = 0; i < (ReadPerThread) / (VecSize); i++)                                     \
//         {                                                                                                           \
//             /*assert((ty) * (ReadPerThread) / VecSize + i + (TileM / VecSize) * (j) < TILE_K * TILE_M / VecSize);*/ \
//             reg[i] = smem[(ty) * (ReadPerThread) / VecSize + i + (TileM / VecSize) * (j)];                          \
//         }                                                                                                           \
//     }

// #define LOAD_SMEM_TO_B_REG_VEC(VecSize, ReadPerThread, TileN, reg, smem, tx, j)                               \
//     {                                                                                                         \
//         _Pragma("unroll") for (int i = 0; i < ReadPerThread / VecSize; i++)                                   \
//         {                                                                                                     \
//             /*assert(tx *ReadPerThread / VecSize + i + (TileN / VecSize) * (j) < TILE_K * TileN / VecSize);*/ \
//             reg[i] = smem[(tx * ReadPerThread / VecSize + i /* x cols*/) + (TileN / VecSize) * (j)];          \
//         }                                                                                                     \
//     }

// #define MMA_VEC(AReadPerThread, BReadPerThread, c, a, b)                  \
//     {                                                                     \
//         _Pragma("unroll") for (size_t i = 0; i < AReadPerThread; ++i)     \
//         {                                                                 \
//             _Pragma("unroll") for (size_t j = 0; j < BReadPerThread; ++j) \
//             {                                                             \
//                 (c)[i * BReadPerThread + j] += (a)[i] * (b)[j];           \
//             }                                                             \
//         }                                                                 \
//     }

// // matrix multiplication kernel
// extern "C" __global__ void matmul_blocked2_vec(
//     const type *__restrict__ A, const type *__restrict__ B, type *__restrict__ C,
//     size_t m, size_t n, size_t k, size_t batch_size)
// {
//     // define shared memory
//     __shared__ type_vec a_tile[2][TILE_K * TILE_M / VEC_SIZE];
//     __shared__ type_vec b_tile[2][TILE_K * TILE_N / VEC_SIZE];

//     constexpr const size_t a_read_per_thread = TILE_M * TILE_K / (THREAD_BLOCK_X * THREAD_BLOCK_Y);
//     constexpr const size_t b_read_per_thread = TILE_N * TILE_K / (THREAD_BLOCK_X * THREAD_BLOCK_Y);

//     static_assert(TILE_M % a_read_per_thread == 0, "a_read_per_thread must be multiple of TILE_M");
//     static_assert(TILE_K % b_read_per_thread == 0, "b_read_per_thread must be multiple of TILE_K");
//     static_assert(a_read_per_thread < TILE_K, "a_read_per_thread must be less than TILE_K");
//     static_assert(b_read_per_thread < TILE_K, "b_read_per_thread must be less than TILE_K");

//     type_vec next_data_a[a_read_per_thread / VEC_SIZE];
//     type_vec next_data_b[b_read_per_thread / VEC_SIZE];
//     type_vec a_reg[2][a_read_per_thread / VEC_SIZE]; // must be multiple of VEC_SIZE
//     type_vec b_reg[2][b_read_per_thread / VEC_SIZE]; // must be multiple of VEC_SIZE

//     // // initialize 4x4 accumulator
//     type_vec c_reg[a_read_per_thread][b_read_per_thread / VEC_SIZE] = {{make_float4(0.f, 0.f, 0.f, 0.f)}};

//     type *c_reg_ptr = reinterpret_cast<type *>(c_reg);

//     size_t txa = threadIdx.x % (TILE_K / a_read_per_thread);
//     size_t tya = threadIdx.x / (TILE_K / a_read_per_thread);

//     size_t txb = threadIdx.x % (TILE_N / b_read_per_thread);
//     size_t tyb = threadIdx.x / (TILE_N / b_read_per_thread);

//     LOAD_GMEM_TO_NEXT_A_REG_VEC(VEC_SIZE, a_read_per_thread, TILE_M, TILE_K, A, next_data_a, txa, tya, k, 0);
//     LOAD_GMEM_TO_NEXT_B_REG_VEC(VEC_SIZE, b_read_per_thread, TILE_N, TILE_K, B, next_data_b, txb, tyb, n, 0);

//     int global_idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * (gridDim.x * blockDim.x);

//     STORE_NEXT_A_DATA_TO_SMEM_VEC(type, VEC_SIZE, a_read_per_thread, TILE_M, next_data_a, a_tile[0], txa, tya);
//     STORE_NEXT_B_DATA_TO_SMEM_VEC(VEC_SIZE, b_read_per_thread, TILE_N, next_data_b, b_tile[0], txb, tyb);

//     size_t tx = threadIdx.x % THREAD_BLOCK_X;
//     size_t ty = threadIdx.x / THREAD_BLOCK_X;

//     __syncthreads();
//     LOAD_SMEM_TO_A_REG_VEC(VEC_SIZE, a_read_per_thread, TILE_M, a_reg[0], a_tile[0], ty, 0);
//     LOAD_SMEM_TO_B_REG_VEC(VEC_SIZE, b_read_per_thread, TILE_N, b_reg[0], b_tile[0], tx, 0);

//     int write_stage_idx = 1;
//     int i = 0;
//     do
//     {
//         i += TILE_K;
//         LOAD_GMEM_TO_NEXT_A_REG_VEC(VEC_SIZE, a_read_per_thread, TILE_M, TILE_K, A, next_data_a, txa, tya, k, i);
//         LOAD_GMEM_TO_NEXT_B_REG_VEC(VEC_SIZE, b_read_per_thread, TILE_N, TILE_K, B, next_data_b, txb, tyb, n, i);

//         int load_stage_idx = write_stage_idx ^ 1;
// #pragma unroll
//         for (int j = 0; j < TILE_K - 1; j++)
//         {
//             LOAD_SMEM_TO_A_REG_VEC(VEC_SIZE, a_read_per_thread, TILE_M, a_reg[(j + 1) % 2], a_tile[load_stage_idx], ty, j + 1);
//             LOAD_SMEM_TO_B_REG_VEC(VEC_SIZE, b_read_per_thread, TILE_N, b_reg[(j + 1) % 2], b_tile[load_stage_idx], tx, j + 1);
//             c_reg[0][0].x += a_reg[j % 2][0].x * b_reg[j % 2][0].x;
//             c_reg[0][0].y += a_reg[j % 2][0].x * b_reg[j % 2][0].y;
//             c_reg[0][0].z += a_reg[j % 2][0].x * b_reg[j % 2][0].z;
//             c_reg[0][0].w += a_reg[j % 2][0].x * b_reg[j % 2][0].w;
//             c_reg[0][1].x += a_reg[j % 2][0].x * b_reg[j % 2][1].x;
//             c_reg[0][1].y += a_reg[j % 2][0].x * b_reg[j % 2][1].y;
//             c_reg[0][1].z += a_reg[j % 2][0].x * b_reg[j % 2][1].z;
//             c_reg[0][1].w += a_reg[j % 2][0].x * b_reg[j % 2][1].w;
//             c_reg[1][0].x += a_reg[j % 2][0].y * b_reg[j % 2][0].x;
//             c_reg[1][0].y += a_reg[j % 2][0].y * b_reg[j % 2][0].y;
//             c_reg[1][0].z += a_reg[j % 2][0].y * b_reg[j % 2][0].z;
//             c_reg[1][0].w += a_reg[j % 2][0].y * b_reg[j % 2][0].w;
//             c_reg[1][1].x += a_reg[j % 2][0].y * b_reg[j % 2][1].x;
//             c_reg[1][1].y += a_reg[j % 2][0].y * b_reg[j % 2][1].y;
//             c_reg[1][1].z += a_reg[j % 2][0].y * b_reg[j % 2][1].z;
//             c_reg[1][1].w += a_reg[j % 2][0].y * b_reg[j % 2][1].w;
//             c_reg[2][0].x += a_reg[j % 2][0].z * b_reg[j % 2][0].x;
//             c_reg[2][0].y += a_reg[j % 2][0].z * b_reg[j % 2][0].y;
//             c_reg[2][0].z += a_reg[j % 2][0].z * b_reg[j % 2][0].z;
//             c_reg[2][0].w += a_reg[j % 2][0].z * b_reg[j % 2][0].w;
//             c_reg[2][1].x += a_reg[j % 2][0].z * b_reg[j % 2][1].x;
//             c_reg[2][1].y += a_reg[j % 2][0].z * b_reg[j % 2][1].y;
//             c_reg[2][1].z += a_reg[j % 2][0].z * b_reg[j % 2][1].z;
//             c_reg[2][1].w += a_reg[j % 2][0].z * b_reg[j % 2][1].w;
//             c_reg[3][0].x += a_reg[j % 2][0].w * b_reg[j % 2][0].x;
//             c_reg[3][0].y += a_reg[j % 2][0].w * b_reg[j % 2][0].y;
//             c_reg[3][0].z += a_reg[j % 2][0].w * b_reg[j % 2][0].z;
//             c_reg[3][0].w += a_reg[j % 2][0].w * b_reg[j % 2][0].w;
//             c_reg[3][1].x += a_reg[j % 2][0].w * b_reg[j % 2][1].x;
//             c_reg[3][1].y += a_reg[j % 2][0].w * b_reg[j % 2][1].y;
//             c_reg[3][1].z += a_reg[j % 2][0].w * b_reg[j % 2][1].z;
//             c_reg[3][1].w += a_reg[j % 2][0].w * b_reg[j % 2][1].w;
//             c_reg[4][0].x += a_reg[j % 2][1].x * b_reg[j % 2][0].x;
//             c_reg[4][0].y += a_reg[j % 2][1].x * b_reg[j % 2][0].y;
//             c_reg[4][0].z += a_reg[j % 2][1].x * b_reg[j % 2][0].z;
//             c_reg[4][0].w += a_reg[j % 2][1].x * b_reg[j % 2][0].w;
//             c_reg[4][1].x += a_reg[j % 2][1].x * b_reg[j % 2][1].x;
//             c_reg[4][1].y += a_reg[j % 2][1].x * b_reg[j % 2][1].y;
//             c_reg[4][1].z += a_reg[j % 2][1].x * b_reg[j % 2][1].z;
//             c_reg[4][1].w += a_reg[j % 2][1].x * b_reg[j % 2][1].w;
//             c_reg[5][0].x += a_reg[j % 2][1].y * b_reg[j % 2][0].x;
//             c_reg[5][0].y += a_reg[j % 2][1].y * b_reg[j % 2][0].y;
//             c_reg[5][0].z += a_reg[j % 2][1].y * b_reg[j % 2][0].z;
//             c_reg[5][0].w += a_reg[j % 2][1].y * b_reg[j % 2][0].w;
//             c_reg[5][1].x += a_reg[j % 2][1].y * b_reg[j % 2][1].x;
//             c_reg[5][1].y += a_reg[j % 2][1].y * b_reg[j % 2][1].y;
//             c_reg[5][1].z += a_reg[j % 2][1].y * b_reg[j % 2][1].z;
//             c_reg[5][1].w += a_reg[j % 2][1].y * b_reg[j % 2][1].w;
//             c_reg[6][0].x += a_reg[j % 2][1].z * b_reg[j % 2][0].x;
//             c_reg[6][0].y += a_reg[j % 2][1].z * b_reg[j % 2][0].y;
//             c_reg[6][0].z += a_reg[j % 2][1].z * b_reg[j % 2][0].z;
//             c_reg[6][0].w += a_reg[j % 2][1].z * b_reg[j % 2][0].w;
//             c_reg[6][1].x += a_reg[j % 2][1].z * b_reg[j % 2][1].x;
//             c_reg[6][1].y += a_reg[j % 2][1].z * b_reg[j % 2][1].y;
//             c_reg[6][1].z += a_reg[j % 2][1].z * b_reg[j % 2][1].z;
//             c_reg[6][1].w += a_reg[j % 2][1].z * b_reg[j % 2][1].w;
//             c_reg[7][0].x += a_reg[j % 2][1].w * b_reg[j % 2][0].x;
//             c_reg[7][0].y += a_reg[j % 2][1].w * b_reg[j % 2][0].y;
//             c_reg[7][0].z += a_reg[j % 2][1].w * b_reg[j % 2][0].z;
//             c_reg[7][0].w += a_reg[j % 2][1].w * b_reg[j % 2][0].w;
//             c_reg[7][1].x += a_reg[j % 2][1].w * b_reg[j % 2][1].x;
//             c_reg[7][1].y += a_reg[j % 2][1].w * b_reg[j % 2][1].y;
//             c_reg[7][1].z += a_reg[j % 2][1].w * b_reg[j % 2][1].z;
//             c_reg[7][1].w += a_reg[j % 2][1].w * b_reg[j % 2][1].w;
//         }

//         if (i < k)
//         {
//             STORE_NEXT_A_DATA_TO_SMEM_VEC(type, VEC_SIZE, a_read_per_thread, TILE_M, next_data_a, a_tile[write_stage_idx], txa, tya);
//             STORE_NEXT_B_DATA_TO_SMEM_VEC(VEC_SIZE, b_read_per_thread, TILE_N, next_data_b, b_tile[write_stage_idx], txb, tyb);
//             __syncthreads();
//             write_stage_idx ^= 1;
//         }

//         LOAD_SMEM_TO_A_REG_VEC(VEC_SIZE, a_read_per_thread, TILE_M, a_reg[0], a_tile[load_stage_idx ^ 1], ty, 0);
//         LOAD_SMEM_TO_B_REG_VEC(VEC_SIZE, b_read_per_thread, TILE_N, b_reg[0], b_tile[load_stage_idx ^ 1], tx, 0);
//             c_reg[0][0].x += a_reg[1][0].x * b_reg[1][0].x;
//             c_reg[0][0].y += a_reg[1][0].x * b_reg[1][0].y;
//             c_reg[0][0].z += a_reg[1][0].x * b_reg[1][0].z;
//             c_reg[0][0].w += a_reg[1][0].x * b_reg[1][0].w;
//             c_reg[0][1].x += a_reg[1][0].x * b_reg[1][1].x;
//             c_reg[0][1].y += a_reg[1][0].x * b_reg[1][1].y;
//             c_reg[0][1].z += a_reg[1][0].x * b_reg[1][1].z;
//             c_reg[0][1].w += a_reg[1][0].x * b_reg[1][1].w;
//             c_reg[1][0].x += a_reg[1][0].y * b_reg[1][0].x;
//             c_reg[1][0].y += a_reg[1][0].y * b_reg[1][0].y;
//             c_reg[1][0].z += a_reg[1][0].y * b_reg[1][0].z;
//             c_reg[1][0].w += a_reg[1][0].y * b_reg[1][0].w;
//             c_reg[1][1].x += a_reg[1][0].y * b_reg[1][1].x;
//             c_reg[1][1].y += a_reg[1][0].y * b_reg[1][1].y;
//             c_reg[1][1].z += a_reg[1][0].y * b_reg[1][1].z;
//             c_reg[1][1].w += a_reg[1][0].y * b_reg[1][1].w;
//             c_reg[2][0].x += a_reg[1][0].z * b_reg[1][0].x;
//             c_reg[2][0].y += a_reg[1][0].z * b_reg[1][0].y;
//             c_reg[2][0].z += a_reg[1][0].z * b_reg[1][0].z;
//             c_reg[2][0].w += a_reg[1][0].z * b_reg[1][0].w;
//             c_reg[2][1].x += a_reg[1][0].z * b_reg[1][1].x;
//             c_reg[2][1].y += a_reg[1][0].z * b_reg[1][1].y;
//             c_reg[2][1].z += a_reg[1][0].z * b_reg[1][1].z;
//             c_reg[2][1].w += a_reg[1][0].z * b_reg[1][1].w;
//             c_reg[3][0].x += a_reg[1][0].w * b_reg[1][0].x;
//             c_reg[3][0].y += a_reg[1][0].w * b_reg[1][0].y;
//             c_reg[3][0].z += a_reg[1][0].w * b_reg[1][0].z;
//             c_reg[3][0].w += a_reg[1][0].w * b_reg[1][0].w;
//             c_reg[3][1].x += a_reg[1][0].w * b_reg[1][1].x;
//             c_reg[3][1].y += a_reg[1][0].w * b_reg[1][1].y;
//             c_reg[3][1].z += a_reg[1][0].w * b_reg[1][1].z;
//             c_reg[3][1].w += a_reg[1][0].w * b_reg[1][1].w;
//             c_reg[4][0].x += a_reg[1][1].x * b_reg[1][0].x;
//             c_reg[4][0].y += a_reg[1][1].x * b_reg[1][0].y;
//             c_reg[4][0].z += a_reg[1][1].x * b_reg[1][0].z;
//             c_reg[4][0].w += a_reg[1][1].x * b_reg[1][0].w;
//             c_reg[4][1].x += a_reg[1][1].x * b_reg[1][1].x;
//             c_reg[4][1].y += a_reg[1][1].x * b_reg[1][1].y;
//             c_reg[4][1].z += a_reg[1][1].x * b_reg[1][1].z;
//             c_reg[4][1].w += a_reg[1][1].x * b_reg[1][1].w;
//             c_reg[5][0].x += a_reg[1][1].y * b_reg[1][0].x;
//             c_reg[5][0].y += a_reg[1][1].y * b_reg[1][0].y;
//             c_reg[5][0].z += a_reg[1][1].y * b_reg[1][0].z;
//             c_reg[5][0].w += a_reg[1][1].y * b_reg[1][0].w;
//             c_reg[5][1].x += a_reg[1][1].y * b_reg[1][1].x;
//             c_reg[5][1].y += a_reg[1][1].y * b_reg[1][1].y;
//             c_reg[5][1].z += a_reg[1][1].y * b_reg[1][1].z;
//             c_reg[5][1].w += a_reg[1][1].y * b_reg[1][1].w;
//             c_reg[6][0].x += a_reg[1][1].z * b_reg[1][0].x;
//             c_reg[6][0].y += a_reg[1][1].z * b_reg[1][0].y;
//             c_reg[6][0].z += a_reg[1][1].z * b_reg[1][0].z;
//             c_reg[6][0].w += a_reg[1][1].z * b_reg[1][0].w;
//             c_reg[6][1].x += a_reg[1][1].z * b_reg[1][1].x;
//             c_reg[6][1].y += a_reg[1][1].z * b_reg[1][1].y;
//             c_reg[6][1].z += a_reg[1][1].z * b_reg[1][1].z;
//             c_reg[6][1].w += a_reg[1][1].z * b_reg[1][1].w;
//             c_reg[7][0].x += a_reg[1][1].w * b_reg[1][0].x;
//             c_reg[7][0].y += a_reg[1][1].w * b_reg[1][0].y;
//             c_reg[7][0].z += a_reg[1][1].w * b_reg[1][0].z;
//             c_reg[7][0].w += a_reg[1][1].w * b_reg[1][0].w;
//             c_reg[7][1].x += a_reg[1][1].w * b_reg[1][1].x;
//             c_reg[7][1].y += a_reg[1][1].w * b_reg[1][1].y;
//             c_reg[7][1].z += a_reg[1][1].w * b_reg[1][1].z;
//             c_reg[7][1].w += a_reg[1][1].w * b_reg[1][1].w;
//     } while (i < k);

//     float *pC = C + blockIdx.y * TILE_M * n + blockIdx.x * TILE_N + tx * a_read_per_thread + ty * b_read_per_thread * n;
//     // acquire_lock(&lock);

//     // release_lock(&lock);

// #pragma unroll(a_read_per_thread)
//     for (int i = 0; i < a_read_per_thread; i++)
//     {
// #pragma unroll(b_read_per_thread)
//         for (int j = 0; j < b_read_per_thread; j++)
//         {
//             pC[i * n + j] = c_reg_ptr[i * b_read_per_thread + j];
//         }
//     }
// }