#include <cstddef>
#include <cstdio>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
// #include <cub/block/block_reduce.cuh>
#include "cuda_utils.h"

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

constexpr int BLOCK_DIM = 512;
constexpr int WARP_SIZE = 32;
constexpr int BYTES_PER_LDG = 16;
constexpr int WARPS_PER_CTA = BLOCK_DIM / WARP_SIZE;

// template <typename T>
// __inline__ __device__ T warpReduceMaxV2(T val) {
// #pragma unroll
//     for (int mask = 16; mask > 0; mask >>= 1) {
//       val = max(val, __shfl_xor_sync(-1, val, mask, 32));
//     }
//   return val;
// }

// template <typename T>
// __inline__ __device__ T warpReduceSumV2(T val) {
// #pragma unroll
//     for (int mask = 16; mask > 0; mask >>= 1) {
//       val += __shfl_xor_sync(-1, val, mask, 32);
//     }
//   return val;
// }


// template <typename T>
// __inline__ __device__ T blockReduceMaxV2(T val) {
//   static __shared__ T shared[32];
//   int lane = threadIdx.x & 0x1f; // in-warp idx
//   int wid = threadIdx.x >> 5; // warp idx
//   warpReduceMaxV2<T>(val);
//   if (lane == 0){ // record in-warp maxx by warp Idx
//     shared[wid] = *val;
//   }
//   __syncthreads();

//   bool is_mask = threadIdx.x < (blockDim.x / 32.f);
//   *val = is_mask ? shared[lane] : (T)-1e20f;
  
//   val = warpReduceMaxV2<T> (val);
//   return 0.0f;
// }

// template <typename T>
// __inline__ __device__ T blockReduceSumV2(T* val) {
//   static __shared__ T shared[32];
//   int lane = threadIdx.x & 0x1f; // in-warp idx
//   int wid = threadIdx.x >> 5; // warp idx

//   warpReduceSumV2<T>(val);

//   if (lane == 0){ // record in-warp maxx by warp Idx
//     shared[wid] = *val;
//   }
//   __syncthreads();

//   bool is_mask = threadIdx.x < (blockDim.x / 32.f);
//   *val = is_mask ? shared[lane] : (T)(0.0f);

//   warpReduceSumV2<T> (val);
//   // printf("block %d, thread %d Sum: %f \n",blockIdx.x, threadIdx.x, *val);
//   return 0.0f;
// }

__global__ void softmax(float *input, float *output, int M, int N)
{
    int row = blockIdx.x;
    __shared__ float tmp[BLOCK_DIM];
    __shared__ float globalMax;
    __shared__ float globalSum;
    //-----------
    float val = -__FLT_MAX__;
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        val = max(val, input[row * N + i]);
    }
    tmp[threadIdx.x] = val;
    __syncthreads();

    for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalMax = tmp[0];
    }
 
    //-----------

    val = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        val += __expf(input[row * N + i] - globalMax);
    }
    tmp[threadIdx.x] = val;
    __syncthreads();
    for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
    {
        if (threadIdx.x < step)
        {
            tmp[threadIdx.x] += tmp[threadIdx.x + step];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        globalSum = tmp[0];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        output[row * N + i] = __expf(input[row * N + i] - globalMax) * __fdividef(1.0F, globalSum);
    }
}

__global__ void softmax1(float *input, float *output, int M, int N)
{
    int row = blockIdx.x;
    __shared__ float tmp[BLOCK_DIM];
    __shared__ float globalMax;
    __shared__ float globalSum;
    //-----------
    float val = -__FLT_MAX__;

    static __shared__ float shared[32];

    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        val = max(val, input[row * N + i]);
    }
    __syncthreads();
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5; // warp idx
    
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val = max(val, __shfl_xor_sync(-1, val, mask, 32));
    }

    if (lane == 0){ // record in-warp maxx by warp Idx
      shared[wid] = val;
    }

     __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    val = is_mask ? shared[lane] : (float) -__FLT_MAX__;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val = max(val, __shfl_xor_sync(-1, val, mask, 32));
    }
    if(threadIdx.x == 0)
      globalMax = val;
    // for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    // {
    //     val = max(val, input[row * N + i]);
    // }
    // tmp[threadIdx.x] = val;
    // __syncthreads();

    // for (int step = BLOCK_DIM / 2; step > 0; step /= 2)
    // {
    //     if (threadIdx.x < step)
    //     {
    //         tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + step]);
    //     }
    //     __syncthreads();
    // }
    // if (threadIdx.x == 0)
    // {
    //     globalMax = tmp[0];
    // }

    //-----------
    val = 0.0f;
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM){
        val += __expf(input[row * N + i] - globalMax);
    }
    __syncthreads();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val += __shfl_xor_sync(-1, val, mask, 32);
    }

    if (lane == 0){ // record in-warp maxx by warp Idx
      shared[wid] = val;
    }
     __syncthreads();
    is_mask = threadIdx.x < (blockDim.x / 32.f);
    val = is_mask ? shared[lane] : (float) 0.0f;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      val += __shfl_xor_sync(-1, val, mask, 32);
    }
    if(threadIdx.x == 0)
      globalSum = val;

    __syncthreads();
    for (int i = threadIdx.x; i < N; i += BLOCK_DIM)
    {
        output[row * N + i] = __expf(input[row * N + i] - globalMax) * __fdividef(1.0F, globalSum);
    }
}
template <typename T, const int VPT, int N, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void softmax_1(
    const T* input,
    T* output,
    const int M) {
  // We begin by enforcing compile time assertions and setting up compile time constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(N == (N & -N), "N must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = N;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  // Restrictions based on previous section.
  static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

  // We have N elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
  // This, each block processes a chunk of rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= M) {
    return;
  }
  // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
  // row it will read.
  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
  T* o_thread_row_ptr = output + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the first column to start loads.
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  T* o_thread_read_ptr = o_thread_row_ptr + first_elt_read_by_thread;

  // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
  // this can support all powers of 2 up to 16.
  float tmp_r[LDG_PER_THREAD * ELTS_PER_LDG];
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    tmp_r[ii * ELTS_PER_LDG] = thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG];
    tmp_r[ii * ELTS_PER_LDG + 1] = thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + 1];
    tmp_r[ii * ELTS_PER_LDG + 2] = thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + 2];
    tmp_r[ii * ELTS_PER_LDG + 3] = thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + 3];
  }

  // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
  // convert to float afterwards for the exp + sum reduction.
  float thread_max = tmp_r[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, tmp_r[ii]);
  }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    tmp_r[ii] = expf(tmp_r[ii] - thread_max);
    row_sum += tmp_r[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
  }

  // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
  // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
  // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    tmp_r[ii] = tmp_r[ii] * reciprocal_row_sum;
  }

#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {

    o_thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG]     = tmp_r[ii * ELTS_PER_LDG]    ;
    o_thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + 1] = tmp_r[ii * ELTS_PER_LDG + 1];
    o_thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + 2] = tmp_r[ii * ELTS_PER_LDG + 2];
    o_thread_read_ptr[ii * THREADS_PER_ROW * ELTS_PER_LDG + 3] = tmp_r[ii * ELTS_PER_LDG + 3];
  }
}

template<int N>
void cpu_softmax(float *cpu_input, float *cpu_output, int M)
{
    double st, ela;
    st = get_walltime();
    int repeat = 100;

    constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    constexpr int VECs_PER_THREAD = std::max(1, N / (ELTS_PER_LDG * WARP_SIZE));
    constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / N;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
    int num_block = M;
    dim3 block_dim(BLOCK_DIM, 1, 1);
    dim3 grid_dim(num_block, 1, 1);

    float *input, *output;
    cudaMalloc((void **)&input, M * N * sizeof(float));
    cudaMalloc((void **)&output, M * N * sizeof(float));
    cudaMemcpy(input, cpu_input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    int n = N;
     for (int i = 0; i < repeat; i++)
    {
    softmax1<<<grid_dim, block_dim>>>(input, output, M, n);
    // softmax_1<float, VPT, N, WARPS_PER_CTA, BYTES_PER_LDG><<<grid_dim, block_dim>>>(input, output, M);
      // softmax_2<float, N><<<grid_dim, block_dim>>>(input, output, M);
    cudaDeviceSynchronize();
    }
    cudaError_t err = cudaGetLastError();                            
        if (err != cudaSuccess) {                                        
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 
                    __FILE__, __LINE__, cudaGetErrorString(err));       
            exit(EXIT_FAILURE);                                          
        }                   
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(cpu_output, output, M * N, cudaMemcpyDeviceToHost);

    cudaFree(input);
    cudaFree(output);

    ela = get_walltime() - st;

    printf("%d x %d kernel time:%.4f\n",M,N, ker_time * 1000);
}

int main()
{
    float *cpu_input, *cpu_output;
    constexpr int Ms[3] = {32,1024,2048};
    constexpr int Ns[3] = {32,1024,2048};
    for(int i = 0;i < 3;++i){
        int M = Ms[i];
        constexpr int N = Ns[0]; 
        cpu_input = (float *)malloc(M * N * sizeof(float));
        cpu_output = (float *)malloc(M * N * sizeof(float));
        for (int i = 0; i < M * N; i++){
            cpu_input[i] = i % 10;
        }

        cpu_softmax<N>(cpu_input, cpu_output, M);

        for (int i = 0; i < 10; i++){
            printf("%.4e ", cpu_output[i]);
        }
        printf("\n");
        free(cpu_input);
        free(cpu_output);
      }
    
    for(int i = 0;i < 3;++i){
        int M = Ms[i];
        constexpr int N = Ns[1]; 
        cpu_input = (float *)malloc(M * N * sizeof(float));
        cpu_output = (float *)malloc(M * N * sizeof(float));
        for (int i = 0; i < M * N; i++){
            cpu_input[i] = i % 10;
        }

        cpu_softmax<N>(cpu_input, cpu_output, M);

        for (int i = 0; i < 10; i++){
            printf("%.4e ", cpu_output[i]);
        }
        printf("\n");
        free(cpu_input);
        free(cpu_output);
      }

    for(int i = 0;i < 3;++i){
        int M = Ms[i];
        constexpr int N=Ns[2]; 
        cpu_input = (float *)malloc(M * N * sizeof(float));
        cpu_output = (float *)malloc(M * N * sizeof(float));
        for (int i = 0; i < M * N; i++){
            cpu_input[i] = i % 10;
        }

        cpu_softmax<N>(cpu_input, cpu_output, M);

        for (int i = 0; i < 10; i++){
            printf("%.4e ", cpu_output[i]);
        }
        printf("\n");
        free(cpu_input);
        free(cpu_output);
      }
    return 0;
}
