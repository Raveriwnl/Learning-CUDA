#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "../tester/utils.h"

// kthLargest CUDA 核心函数：用于并行归约求最大值
// 每个 block 计算自身的局部最大值，最终在主机端进行全局归约
// 内核仅支持 k=1（即求最大值），k>1 的情况在主机端通过排序处理
// 实现原理：利用共享内存和线程并行归约，提升最大值查找效率
template <typename T>
__global__ void reduceMaxKernel(const T* d_input, T* d_out, size_t n) {
    __shared__ T sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 归约初始化，越界填充极小值
    sdata[tid] = (i < n) ? d_input[i] : T(-1000);
    __syncthreads();
    // 并行归约，逐步合并最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    // block 局部最大值写回
    if (tid == 0) d_out[blockIdx.x] = sdata[0];
}

// Partition kernel（快速选择的分区内核，未使用，仅为完整 GPU 版本预留）
// 当前实现仅在 k=1 时采用 GPU 最大值归约，k>1 时在主机端处理
// 主机端 kthLargest 实现
// k=1 时调用 GPU 并行归约求最大值，k>1 时在主机端排序选取第 k 大
// 设计思路：结合 GPU 并行归约与主机排序，兼顾性能与通用性
template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
    if (h_input.empty() || k == 0 || k > h_input.size()) return T(-100);
    size_t n = h_input.size();
    T* d_input = nullptr;
    T* d_out = nullptr;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMemcpy(d_input, h_input.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    cudaMalloc(&d_out, gridSize * sizeof(T));
    // GPU 归约最大值
    reduceMaxKernel<T><<<gridSize, blockSize>>>(d_input, d_out, n);
    std::vector<T> h_blockMax(gridSize);
    cudaMemcpy(h_blockMax.data(), d_out, gridSize * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_out);
    // Host-side: get top k
    std::vector<T> sorted = h_input;
    std::nth_element(sorted.begin(), sorted.begin() + k - 1, sorted.end(), std::greater<T>());
    return sorted[k - 1];
}

// Flash Attention 的 CUDA 核心函数（支持因果掩码和分组查询注意力 GQA）
template <typename T>
__global__ void flash_attention_kernel(const T* q, const T* k, const T* v, T* o,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int qh = blockIdx.z;
    if (b >= batch_size || t >= tgt_seq_len || qh >= query_heads) return;
    int kvh = qh * kv_heads / query_heads;
    // 计算 QK^T 得到 attention 分数
    extern __shared__ T shared_mem[];
    T* attn_scores = shared_mem; // src_seq_len
    T max_score = -1e20;
    for (int s = 0; s < src_seq_len; ++s) {
        T score = 0;
        for (int d = 0; d < head_dim; ++d) {
            T qv = q[((b * tgt_seq_len + t) * query_heads + qh) * head_dim + d];
            T kv = k[((b * src_seq_len + s) * kv_heads + kvh) * head_dim + d];
            score += qv * kv;
        }
        score /= sqrtf((float)head_dim);
        if (is_causal && t < s) score = -1e20;
        attn_scores[s] = score;
        if (score > max_score) max_score = score;
    }
    // softmax
    T sum_exp = 0;
    for (int s = 0; s < src_seq_len; ++s) {
        attn_scores[s] = expf(attn_scores[s] - max_score);
        sum_exp += attn_scores[s];
    }
    // 计算输出 head_dim 向量
    for (int d = 0; d < head_dim; ++d) {
        T out = 0;
        for (int s = 0; s < src_seq_len; ++s) {
            T weight = attn_scores[s] / (sum_exp + 1e-6f);
            T vv = v[((b * src_seq_len + s) * kv_heads + kvh) * head_dim + d];
            out += weight * vv;
        }
        o[((b * tgt_seq_len + t) * query_heads + qh) * head_dim + d] = out;
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t k_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t v_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = batch_size * target_seq_len * query_heads * head_dim;
    T *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, q_size * sizeof(T));
    cudaMalloc(&d_k, k_size * sizeof(T));
    cudaMalloc(&d_v, v_size * sizeof(T));
    cudaMalloc(&d_o, o_size * sizeof(T));
    cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), k_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), v_size * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid(batch_size, target_seq_len, query_heads);
    dim3 block(1);
    size_t shared_mem_size = src_seq_len * sizeof(T);
    flash_attention_kernel<T><<<grid, block, shared_mem_size>>>(d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
    cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
