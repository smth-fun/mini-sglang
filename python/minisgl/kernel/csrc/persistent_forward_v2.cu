/*
 * Persistent forward kernel using CUDA cooperative groups.
 * Implements the entire decode forward pass for Qwen3-0.6B in ONE kernel.
 * Eliminates ~199 kernel transitions per decode step.
 *
 * Model: Qwen3-0.6B (hardcoded for this exact model)
 * INT8 weights with per-row float32 scales.
 */
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

// Model constants (Qwen3-0.6B: head_dim=128, NOT hidden/num_heads!)
constexpr int HIDDEN = 1024;
constexpr int NUM_LAYERS = 28;
constexpr int NUM_QO = 16;
constexpr int NUM_KV = 8;
constexpr int HEAD_DIM = 128;
constexpr int HALF_DIM = 64;
constexpr int GQA_RATIO = 2;
constexpr int INTER = 3072;
constexpr int QKV_DIM = (NUM_QO + 2 * NUM_KV) * HEAD_DIM;  // 4096
constexpr int QO_DIM = NUM_QO * HEAD_DIM;   // 2048
constexpr int KV_DIM = NUM_KV * HEAD_DIM;   // 1024
constexpr int BLOCK_SEQ = 128;
constexpr int NUM_CHUNKS = 16;
constexpr int THREADS = 128;

// Max shared memory: max(INTER, HIDDEN) * sizeof(float) for GEMV,
// or attention needs: HEAD_DIM*4 + BLOCK_SEQ*4 + BLOCK_SEQ*4 ~ 1.5KB
// So GEMV dominates: INTER * 4 = 12288 bytes
constexpr int SMEM_BYTES = INTER * sizeof(float);  // 12288

struct LayerWeights {
    const bf16* input_norm_w;
    const int8_t* qkv_w;
    const float* qkv_scale;
    const bf16* q_norm_w;
    const bf16* k_norm_w;
    const int8_t* o_w;
    const float* o_scale;
    const bf16* post_norm_w;
    const int8_t* gate_up_w;
    const float* gate_up_scale;
    const int8_t* down_w;
    const float* down_scale;
    bf16* k_cache;
    bf16* v_cache;
};

struct PersistentArgs {
    const bf16* embed_w;
    LayerWeights layers[NUM_LAYERS];
    const bf16* final_norm_w;
    const int8_t* lm_head_w;
    const float* lm_head_scale;
    float eps;
    const float* cos_sin_cache;
    int cos_sin_stride;
    const int* page_indices;
    const int* seq_len_ptr;
    int cache_stride_token;
    float sm_scale;
    const int* input_ids;
    const int* positions;
    const int* out_loc;
    bf16* x_buf;
    bf16* residual_buf_a;  // double-buffered residual
    bf16* residual_buf_b;
    bf16* qkv_buf;
    bf16* attn_out_buf;
    bf16* mlp_hidden_buf;
    float* partial_out_buf;
    float* partial_lse_buf;
    float* logits_buf;
    int vocab_size;
    int num_layers_to_run;  // debug: limit number of layers
    float* debug_buf;       // debug output buffer
    int debug_buf_size;
};

// ============================================================
// Block-wide reductions
// ============================================================

// shared memory for reductions (separate from main smem)
// We use the beginning of extern shared memory for the main data,
// and reserve a small area at the end for reduction workspace.
// Actually, we'll use fixed __shared__ for reductions.

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// These use a dedicated __shared__ workspace
__device__ float block_reduce_sum(float val) {
    __shared__ float rs[32];
    int wid = threadIdx.x / 32, lid = threadIdx.x % 32;
    val = warp_reduce_sum(val);
    if (lid == 0) rs[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lid < (blockDim.x + 31) / 32) ? rs[lid] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    if (threadIdx.x == 0) rs[0] = val;
    __syncthreads();
    return rs[0];
}

__device__ float block_reduce_max(float val) {
    __shared__ float rm[32];
    int wid = threadIdx.x / 32, lid = threadIdx.x % 32;
    val = warp_reduce_max(val);
    if (lid == 0) rm[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lid < (blockDim.x + 31) / 32) ? rm[lid] : -1e30f;
        val = warp_reduce_max(val);
    }
    __syncthreads();
    if (threadIdx.x == 0) rm[0] = val;
    __syncthreads();
    return rm[0];
}

// ============================================================
// Phase: Embedding lookup (block 0 only)
// ============================================================
__device__ void phase_embed(const PersistentArgs* args, int bid) {
    if (bid != 0) return;
    int token_id = args->input_ids[0];
    const bf16* row = args->embed_w + (long long)token_id * HIDDEN;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
        args->x_buf[i] = row[i];
}

// ============================================================
// Phase: Copy x_buf to residual_buf_a (for layer 0)
// ============================================================
__device__ void phase_copy_to_residual(const PersistentArgs* args, int bid) {
    if (bid != 0) return;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
        args->residual_buf_a[i] = args->x_buf[i];
}

// ============================================================
// Phase: RMSNorm + INT8 GEMV (first layer only, no residual add)
// ============================================================
__device__ void phase_norm_gemv_int8(
    const bf16* x, const bf16* norm_w,
    const int8_t* weight, const float* scale,
    bf16* out, int M, int N, float eps,
    int bid, int num_blocks
) {
    extern __shared__ float smem[];

    // Compute RMSNorm
    float ss = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        smem[i] = v;
        ss += v * v;
    }
    __syncthreads();
    ss = block_reduce_sum(ss);
    float rms_inv = rsqrtf(ss / M + eps);

    for (int i = threadIdx.x; i < M; i += blockDim.x)
        smem[i] = smem[i] * rms_inv * __bfloat162float(norm_w[i]);
    __syncthreads();

    // GEMV
    for (int row = bid; row < N; row += num_blocks) {
        float acc = 0.0f;
        const int8_t* w = weight + (long long)row * M;
        for (int m = threadIdx.x; m < M; m += blockDim.x)
            acc += (float)w[m] * smem[m];
        acc = block_reduce_sum(acc);
        if (threadIdx.x == 0)
            out[row] = __float2bfloat16(acc * scale[row]);
    }
}

// ============================================================
// Phase: Fused Add + RMSNorm + INT8 GEMV
// ============================================================
__device__ void phase_add_norm_gemv_int8(
    const bf16* x, const bf16* residual, const bf16* norm_w,
    const int8_t* weight, const float* scale,
    bf16* out, bf16* res_out,
    int M, int N, float eps,
    int bid, int num_blocks
) {
    extern __shared__ float smem[];

    float ss = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float v = __bfloat162float(x[i]) + __bfloat162float(residual[i]);
        smem[i] = v;
        ss += v * v;
        if (bid == 0) res_out[i] = __float2bfloat16(v);
    }
    __syncthreads();
    ss = block_reduce_sum(ss);
    float rms_inv = rsqrtf(ss / M + eps);

    for (int i = threadIdx.x; i < M; i += blockDim.x)
        smem[i] = smem[i] * rms_inv * __bfloat162float(norm_w[i]);
    __syncthreads();

    for (int row = bid; row < N; row += num_blocks) {
        float acc = 0.0f;
        const int8_t* w = weight + (long long)row * M;
        for (int m = threadIdx.x; m < M; m += blockDim.x)
            acc += (float)w[m] * smem[m];
        acc = block_reduce_sum(acc);
        if (threadIdx.x == 0)
            out[row] = __float2bfloat16(acc * scale[row]);
    }
}

// ============================================================
// Phase: Fused Add + RMSNorm + INT8 gate_up GEMV + SiLU
// ============================================================
__device__ void phase_add_norm_gateup_silu_int8(
    const bf16* x, const bf16* residual, const bf16* norm_w,
    const int8_t* weight, const float* scale,
    bf16* out, bf16* res_out,
    int M, int N_half, float eps,
    int bid, int num_blocks
) {
    extern __shared__ float smem[];

    float ss = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float v = __bfloat162float(x[i]) + __bfloat162float(residual[i]);
        smem[i] = v;
        ss += v * v;
        if (bid == 0) res_out[i] = __float2bfloat16(v);
    }
    __syncthreads();
    ss = block_reduce_sum(ss);
    float rms_inv = rsqrtf(ss / M + eps);

    for (int i = threadIdx.x; i < M; i += blockDim.x)
        smem[i] = smem[i] * rms_inv * __bfloat162float(norm_w[i]);
    __syncthreads();

    for (int row = bid; row < N_half; row += num_blocks) {
        float acc_g = 0.0f, acc_u = 0.0f;
        const int8_t* wg = weight + (long long)row * M;
        const int8_t* wu = weight + (long long)(row + N_half) * M;
        for (int m = threadIdx.x; m < M; m += blockDim.x) {
            float s = smem[m];
            acc_g += (float)wg[m] * s;
            acc_u += (float)wu[m] * s;
        }
        acc_g = block_reduce_sum(acc_g);
        acc_u = block_reduce_sum(acc_u);
        if (threadIdx.x == 0) {
            acc_g *= scale[row];
            acc_u *= scale[row + N_half];
            float silu = acc_g / (1.0f + expf(-acc_g));
            out[row] = __float2bfloat16(silu * acc_u);
        }
    }
}

// ============================================================
// Phase: Plain INT8 GEMV (for O-proj, down_proj)
// ============================================================
__device__ void phase_gemv_int8(
    const bf16* x_in, const int8_t* weight, const float* scale,
    bf16* out, int M, int N,
    int bid, int num_blocks
) {
    extern __shared__ float smem[];
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        smem[i] = __bfloat162float(x_in[i]);
    __syncthreads();

    for (int row = bid; row < N; row += num_blocks) {
        float acc = 0.0f;
        const int8_t* w = weight + (long long)row * M;
        for (int m = threadIdx.x; m < M; m += blockDim.x)
            acc += (float)w[m] * smem[m];
        acc = block_reduce_sum(acc);
        if (threadIdx.x == 0)
            out[row] = __float2bfloat16(acc * scale[row]);
    }
}

// ============================================================
// Phase: QK Norm + RoPE + KV Store
// Each block handles one head (Q or K). Only first 24 blocks active.
// All 128 threads participate in __syncthreads(); only first 64 do work.
// ============================================================
__device__ void phase_qk_norm_rope_store(
    bf16* qkv, const bf16* q_nw, const bf16* k_nw,
    const float* cos_sin_cache, int cos_sin_stride,
    const int* positions, const int* out_loc,
    bf16* k_cache, bf16* v_cache,
    int cache_stride_token, float eps,
    int bid
) {
    int total_heads = NUM_QO + NUM_KV;  // 24
    if (bid >= total_heads) return;  // entire block returns - OK

    int pos = positions[0];
    int oloc = out_loc[0];
    const float* cs = cos_sin_cache + pos * cos_sin_stride;

    int tid = threadIdx.x;
    // With HEAD_DIM=128 and THREADS=128, all threads are active

    __shared__ float ws_qk[4];

    if (bid < NUM_QO) {
        // Q head
        bf16* base = qkv + bid * HEAD_DIM;
        float val = __bfloat162float(base[tid]);
        float sq = val * val;

        sq = warp_reduce_sum(sq);
        int wid = tid / 32, lid = tid % 32;
        if (lid == 0) ws_qk[wid] = sq;
        __syncthreads();
        if (tid == 0) {
            float total_ss = ws_qk[0] + ws_qk[1] + ws_qk[2] + ws_qk[3];
            ws_qk[0] = rsqrtf(total_ss / HEAD_DIM + eps);
        }
        __syncthreads();
        float rrms = ws_qk[0];

        float nw = __bfloat162float(q_nw[tid]);
        float normed = val * rrms * nw;

        if (tid < HALF_DIM) {
            float cos_v = cs[tid];
            float sin_v = cs[HALF_DIM + tid];
            float hi_val = __bfloat162float(base[tid + HALF_DIM]);
            float hi_nw = __bfloat162float(q_nw[tid + HALF_DIM]);
            float hi_normed = hi_val * rrms * hi_nw;
            base[tid] = __float2bfloat16(normed * cos_v - hi_normed * sin_v);
            base[tid + HALF_DIM] = __float2bfloat16(hi_normed * cos_v + normed * sin_v);
        }
    } else {
        // K head
        int kv_idx = bid - NUM_QO;
        bf16* k_base = qkv + QO_DIM + kv_idx * HEAD_DIM;

        float val = __bfloat162float(k_base[tid]);
        float sq = val * val;

        sq = warp_reduce_sum(sq);
        int wid = tid / 32, lid = tid % 32;
        if (lid == 0) ws_qk[wid] = sq;
        __syncthreads();
        if (tid == 0) {
            float total_ss = ws_qk[0] + ws_qk[1] + ws_qk[2] + ws_qk[3];
            ws_qk[0] = rsqrtf(total_ss / HEAD_DIM + eps);
        }
        __syncthreads();
        float rrms = ws_qk[0];

        float nw = __bfloat162float(k_nw[tid]);
        float normed = val * rrms * nw;

        if (tid < HALF_DIM) {
            float cos_v = cs[tid];
            float sin_v = cs[HALF_DIM + tid];
            float hi_val = __bfloat162float(k_base[tid + HALF_DIM]);
            float hi_nw = __bfloat162float(k_nw[tid + HALF_DIM]);
            float hi_normed = hi_val * rrms * hi_nw;
            k_base[tid] = __float2bfloat16(normed * cos_v - hi_normed * sin_v);
            k_base[tid + HALF_DIM] = __float2bfloat16(hi_normed * cos_v + normed * sin_v);

            long long c_off = (long long)oloc * cache_stride_token + kv_idx * HEAD_DIM;
            k_cache[c_off + tid] = __float2bfloat16(normed * cos_v - hi_normed * sin_v);
            k_cache[c_off + tid + HALF_DIM] = __float2bfloat16(hi_normed * cos_v + normed * sin_v);
        }

        // Store V to cache (all 128 threads)
        bf16* v_src = qkv + QO_DIM + KV_DIM + kv_idx * HEAD_DIM;
        long long vc_off = (long long)oloc * cache_stride_token + kv_idx * HEAD_DIM;
        v_cache[vc_off + tid] = v_src[tid];
    }
}

// ============================================================
// Phase: Split-K Attention
// ============================================================
__device__ void phase_splitk_attention(
    const bf16* qkv, const bf16* k_cache, const bf16* v_cache,
    const int* page_indices, const int* seq_len_ptr,
    float sm_scale, int cache_stride_token,
    float* partial_out, float* partial_lse,
    int bid, int num_blocks
) {
    int total_work = NUM_QO * NUM_CHUNKS;  // 256

    for (int work_id = bid; work_id < total_work; work_id += num_blocks) {
        int qo_head = work_id / NUM_CHUNKS;
        int chunk_id = work_id % NUM_CHUNKS;
        int kv_head = qo_head / GQA_RATIO;
        int seq_len = seq_len_ptr[0];
        int seq_start = chunk_id * BLOCK_SEQ;

        // Load Q into shared memory
        __shared__ float s_q[HEAD_DIM];
        if (threadIdx.x < HEAD_DIM)
            s_q[threadIdx.x] = __bfloat162float(qkv[qo_head * HEAD_DIM + threadIdx.x]);
        __syncthreads();

        // Step 1: Compute scores (1 thread per position, 128 threads for 128 positions)
        __shared__ float s_scores[BLOCK_SEQ];

        float my_score = -1e30f;
        int my_pos = seq_start + threadIdx.x;
        bool valid = (threadIdx.x < BLOCK_SEQ) && (my_pos < seq_len);

        if (valid) {
            int page = page_indices[my_pos];
            long long k_off = (long long)page * cache_stride_token + kv_head * HEAD_DIM;
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                score += __bfloat162float(k_cache[k_off + d]) * s_q[d];
            my_score = score * sm_scale;
        }
        s_scores[threadIdx.x] = my_score;
        __syncthreads();

        // Step 2: Find max score (block reduce)
        float m = block_reduce_max(valid ? my_score : -1e30f);

        // Step 3: Compute exp scores and sum
        float my_exp = valid ? expf(my_score - m) : 0.0f;
        float l = block_reduce_sum(my_exp);

        // Step 4: Compute weighted V (thread d handles V dimension d)
        // Reuse s_scores to store exp_scores
        s_scores[threadIdx.x] = my_exp;
        __syncthreads();

        if (threadIdx.x < HEAD_DIM) {
            int d = threadIdx.x;
            float acc = 0.0f;
            for (int s = 0; s < BLOCK_SEQ; s++) {
                int pos = seq_start + s;
                if (pos >= seq_len) break;
                float w = s_scores[s];
                int page = page_indices[pos];
                long long v_off = (long long)page * cache_stride_token + kv_head * HEAD_DIM + d;
                acc += w * __bfloat162float(v_cache[v_off]);
            }

            // Normalize and store
            float safe_l = (l > 0.0f) ? l : 1.0f;
            int out_off = qo_head * NUM_CHUNKS * HEAD_DIM + chunk_id * HEAD_DIM + d;
            partial_out[out_off] = acc / safe_l;
        }

        // Store LSE
        if (threadIdx.x == 0) {
            int lse_off = qo_head * NUM_CHUNKS + chunk_id;
            partial_lse[lse_off] = (l > 0.0f) ? (m + logf(l)) : -1e30f;
        }
        __syncthreads();  // ensure all writes complete before next work item
    }
}

// ============================================================
// Phase: Attention Reduce
// ============================================================
__device__ void phase_reduce_attention(
    const float* partial_out, const float* partial_lse,
    bf16* out, int bid
) {
    if (bid >= NUM_QO) return;
    int qo_head = bid;

    // Use thread 0 for simplicity (only 16 chunks × 64 dims to process)
    if (threadIdx.x == 0) {
        float m_global = -1e30f;
        for (int c = 0; c < NUM_CHUNKS; c++) {
            float lse = partial_lse[qo_head * NUM_CHUNKS + c];
            m_global = fmaxf(m_global, lse);
        }

        float total_w = 0.0f;
        float acc[HEAD_DIM] = {};
        for (int c = 0; c < NUM_CHUNKS; c++) {
            float lse = partial_lse[qo_head * NUM_CHUNKS + c];
            float w = expf(lse - m_global);
            total_w += w;
            int off = qo_head * NUM_CHUNKS * HEAD_DIM + c * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++)
                acc[d] += partial_out[off + d] * w;
        }

        for (int d = 0; d < HEAD_DIM; d++)
            out[qo_head * HEAD_DIM + d] = __float2bfloat16(acc[d] / total_w);
    }
}

// ============================================================
// Phase: Final Add + RMSNorm + lm_head GEMV (float32 output)
// ============================================================
__device__ void phase_final_norm_lmhead_int8(
    const bf16* x, const bf16* residual, const bf16* norm_w,
    const int8_t* weight, const float* scale,
    float* out, int M, int N, float eps,
    int bid, int num_blocks
) {
    extern __shared__ float smem[];

    float ss = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        float v = __bfloat162float(x[i]) + __bfloat162float(residual[i]);
        smem[i] = v;
        ss += v * v;
    }
    __syncthreads();
    ss = block_reduce_sum(ss);
    float rms_inv = rsqrtf(ss / M + eps);

    for (int i = threadIdx.x; i < M; i += blockDim.x)
        smem[i] = smem[i] * rms_inv * __bfloat162float(norm_w[i]);
    __syncthreads();

    for (int row = bid; row < N; row += num_blocks) {
        float acc = 0.0f;
        const int8_t* w = weight + (long long)row * M;
        for (int m = threadIdx.x; m < M; m += blockDim.x)
            acc += (float)w[m] * smem[m];
        acc = block_reduce_sum(acc);
        if (threadIdx.x == 0)
            out[row] = acc * scale[row];  // float32 logits
    }
}

// ============================================================
// MAIN PERSISTENT KERNEL
// ============================================================
__global__ void persistent_forward_kernel(PersistentArgs* args) {
    auto grid = cg::this_grid();
    int bid = blockIdx.x;
    int nblk = gridDim.x;

    // === Embedding ===
    phase_embed(args, bid);
    grid.sync();

    // Debug: save embedding in debug_buf[0..1023]
    if (bid == 0 && args->debug_buf) {
        for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
            args->debug_buf[i] = __bfloat162float(args->x_buf[i]);
    }

    // === Copy embedding to residual (for layer 0) ===
    phase_copy_to_residual(args, bid);
    grid.sync();

    // === Process layers ===
    // Double-buffered residual: read from res_rd, write to res_wr
    bf16* res_rd = args->residual_buf_a;  // starts with embedding
    bf16* res_wr = args->residual_buf_b;

    int layers_to_run = args->num_layers_to_run > 0 ? args->num_layers_to_run : NUM_LAYERS;
    for (int L = 0; L < layers_to_run; L++) {
        const LayerWeights& lw = args->layers[L];

        // Phase 1: Input norm + QKV GEMV
        if (L == 0) {
            // First layer: RMSNorm only (no residual add)
            phase_norm_gemv_int8(
                args->x_buf, lw.input_norm_w,
                lw.qkv_w, lw.qkv_scale, args->qkv_buf,
                HIDDEN, QKV_DIM, args->eps, bid, nblk);
            // res_rd still = embedding, no swap needed
        } else {
            // Reads res_rd, writes new residual to res_wr (different buffer!)
            phase_add_norm_gemv_int8(
                args->x_buf, res_rd, lw.input_norm_w,
                lw.qkv_w, lw.qkv_scale, args->qkv_buf, res_wr,
                HIDDEN, QKV_DIM, args->eps, bid, nblk);
            // Swap: now res_rd = freshly written residual
            bf16* tmp = res_rd; res_rd = res_wr; res_wr = tmp;
        }
        grid.sync();

        // Debug: save QKV after layer 0 in debug_buf[1024..3071]
        if (L == 0 && bid == 0 && args->debug_buf) {
            for (int i = threadIdx.x; i < QKV_DIM; i += blockDim.x)
                args->debug_buf[1024 + i] = __bfloat162float(args->qkv_buf[i]);
        }

        // Phase 2: QK norm + RoPE + KV store
        phase_qk_norm_rope_store(
            args->qkv_buf, lw.q_norm_w, lw.k_norm_w,
            args->cos_sin_cache, args->cos_sin_stride,
            args->positions, args->out_loc,
            lw.k_cache, lw.v_cache,
            args->cache_stride_token, args->eps, bid);
        grid.sync();

        // Phase 3: Split-K attention
        phase_splitk_attention(
            args->qkv_buf, lw.k_cache, lw.v_cache,
            args->page_indices, args->seq_len_ptr,
            args->sm_scale, args->cache_stride_token,
            args->partial_out_buf, args->partial_lse_buf,
            bid, nblk);
        grid.sync();

        // Phase 4: Attention reduce
        phase_reduce_attention(
            args->partial_out_buf, args->partial_lse_buf,
            args->attn_out_buf, bid);
        grid.sync();

        // Phase 5: O-proj GEMV (input=QO_DIM=2048, output=HIDDEN=1024)
        phase_gemv_int8(
            args->attn_out_buf, lw.o_w, lw.o_scale,
            args->x_buf, QO_DIM, HIDDEN, bid, nblk);
        grid.sync();

        // Phase 6: Post-attn norm + gate_up GEMV + SiLU
        // Reads res_rd, writes new residual to res_wr (different buffer!)
        phase_add_norm_gateup_silu_int8(
            args->x_buf, res_rd, lw.post_norm_w,
            lw.gate_up_w, lw.gate_up_scale,
            args->mlp_hidden_buf, res_wr,
            HIDDEN, INTER, args->eps, bid, nblk);
        // Swap
        { bf16* tmp = res_rd; res_rd = res_wr; res_wr = tmp; }
        grid.sync();

        // Phase 7: down_proj GEMV
        phase_gemv_int8(
            args->mlp_hidden_buf, lw.down_w, lw.down_scale,
            args->x_buf, INTER, HIDDEN, bid, nblk);

        if (L < NUM_LAYERS - 1)
            grid.sync();
    }
    grid.sync();

    // === Final norm + lm_head ===
    phase_final_norm_lmhead_int8(
        args->x_buf, res_rd, args->final_norm_w,
        args->lm_head_w, args->lm_head_scale, args->logits_buf,
        HIDDEN, args->vocab_size, args->eps, bid, nblk);
}


// ============================================================
// Host functions
// ============================================================

static PersistentArgs* d_args = nullptr;
static int g_num_blocks = 0;

torch::Tensor init_persistent_forward(
    torch::Tensor embed_w,
    std::vector<torch::Tensor> input_norm_w,
    std::vector<torch::Tensor> qkv_w,
    std::vector<torch::Tensor> qkv_scale,
    std::vector<torch::Tensor> q_norm_w,
    std::vector<torch::Tensor> k_norm_w,
    std::vector<torch::Tensor> o_w,
    std::vector<torch::Tensor> o_scale,
    std::vector<torch::Tensor> post_norm_w,
    std::vector<torch::Tensor> gate_up_w,
    std::vector<torch::Tensor> gate_up_scale,
    std::vector<torch::Tensor> down_w,
    std::vector<torch::Tensor> down_scale,
    std::vector<torch::Tensor> k_cache,
    std::vector<torch::Tensor> v_cache,
    torch::Tensor final_norm_w,
    torch::Tensor lm_head_w,
    torch::Tensor lm_head_scale,
    float eps,
    torch::Tensor cos_sin_cache,
    int cos_sin_stride,
    torch::Tensor page_indices,
    torch::Tensor seq_len_tensor,
    int cache_stride_token,
    float sm_scale,
    torch::Tensor input_ids_buf,
    torch::Tensor positions_buf,
    torch::Tensor out_loc_buf,
    torch::Tensor x_buf,
    torch::Tensor residual_buf_a,
    torch::Tensor residual_buf_b,
    torch::Tensor qkv_buf,
    torch::Tensor attn_out_buf,
    torch::Tensor mlp_hidden_buf,
    torch::Tensor partial_out_buf,
    torch::Tensor partial_lse_buf,
    torch::Tensor logits_buf,
    int vocab_size
) {
    PersistentArgs h;
    h.embed_w = reinterpret_cast<const bf16*>(embed_w.data_ptr());
    for (int i = 0; i < NUM_LAYERS; i++) {
        h.layers[i].input_norm_w = reinterpret_cast<const bf16*>(input_norm_w[i].data_ptr());
        h.layers[i].qkv_w = qkv_w[i].data_ptr<int8_t>();
        h.layers[i].qkv_scale = qkv_scale[i].data_ptr<float>();
        h.layers[i].q_norm_w = reinterpret_cast<const bf16*>(q_norm_w[i].data_ptr());
        h.layers[i].k_norm_w = reinterpret_cast<const bf16*>(k_norm_w[i].data_ptr());
        h.layers[i].o_w = o_w[i].data_ptr<int8_t>();
        h.layers[i].o_scale = o_scale[i].data_ptr<float>();
        h.layers[i].post_norm_w = reinterpret_cast<const bf16*>(post_norm_w[i].data_ptr());
        h.layers[i].gate_up_w = gate_up_w[i].data_ptr<int8_t>();
        h.layers[i].gate_up_scale = gate_up_scale[i].data_ptr<float>();
        h.layers[i].down_w = down_w[i].data_ptr<int8_t>();
        h.layers[i].down_scale = down_scale[i].data_ptr<float>();
        h.layers[i].k_cache = reinterpret_cast<bf16*>(k_cache[i].data_ptr());
        h.layers[i].v_cache = reinterpret_cast<bf16*>(v_cache[i].data_ptr());
    }
    h.final_norm_w = reinterpret_cast<const bf16*>(final_norm_w.data_ptr());
    h.lm_head_w = lm_head_w.data_ptr<int8_t>();
    h.lm_head_scale = lm_head_scale.data_ptr<float>();
    h.eps = eps;
    h.cos_sin_cache = cos_sin_cache.data_ptr<float>();
    h.cos_sin_stride = cos_sin_stride;
    h.page_indices = page_indices.data_ptr<int>();
    h.seq_len_ptr = seq_len_tensor.data_ptr<int>();
    h.cache_stride_token = cache_stride_token;
    h.sm_scale = sm_scale;
    h.input_ids = input_ids_buf.data_ptr<int>();
    h.positions = positions_buf.data_ptr<int>();
    h.out_loc = out_loc_buf.data_ptr<int>();
    h.x_buf = reinterpret_cast<bf16*>(x_buf.data_ptr());
    h.residual_buf_a = reinterpret_cast<bf16*>(residual_buf_a.data_ptr());
    h.residual_buf_b = reinterpret_cast<bf16*>(residual_buf_b.data_ptr());
    h.qkv_buf = reinterpret_cast<bf16*>(qkv_buf.data_ptr());
    h.attn_out_buf = reinterpret_cast<bf16*>(attn_out_buf.data_ptr());
    h.mlp_hidden_buf = reinterpret_cast<bf16*>(mlp_hidden_buf.data_ptr());
    h.partial_out_buf = partial_out_buf.data_ptr<float>();
    h.partial_lse_buf = partial_lse_buf.data_ptr<float>();
    h.logits_buf = logits_buf.data_ptr<float>();
    h.vocab_size = vocab_size;
    h.num_layers_to_run = 0;  // 0 = all layers
    h.debug_buf = nullptr;
    h.debug_buf_size = 0;

    if (d_args) cudaFree(d_args);
    cudaMalloc(&d_args, sizeof(PersistentArgs));
    cudaMemcpy(d_args, &h, sizeof(PersistentArgs), cudaMemcpyHostToDevice);

    int device;
    cudaGetDevice(&device);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
    g_num_blocks = numSMs;  // 1 block per SM for best grid.sync()

    auto info = torch::zeros({2}, torch::kInt32);
    info[0] = g_num_blocks;
    info[1] = numSMs;
    return info;
}

void set_debug_buffer(torch::Tensor debug_buf) {
    assert(d_args != nullptr);
    PersistentArgs h;
    cudaMemcpy(&h, d_args, sizeof(PersistentArgs), cudaMemcpyDeviceToHost);
    h.debug_buf = debug_buf.data_ptr<float>();
    h.debug_buf_size = debug_buf.numel();
    cudaMemcpy(d_args, &h, sizeof(PersistentArgs), cudaMemcpyHostToDevice);
}

void launch_persistent_forward() {
    assert(d_args != nullptr && "Must call init_persistent_forward first");
    void* kernel_args[] = {(void*)&d_args};
    // Use PyTorch's current CUDA stream so we respect stream ordering
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)persistent_forward_kernel,
        g_num_blocks, THREADS, kernel_args, SMEM_BYTES, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Cooperative kernel launch failed: ") + cudaGetErrorString(err));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_persistent_forward", &init_persistent_forward);
    m.def("set_debug_buffer", &set_debug_buffer);
    m.def("launch_persistent_forward", &launch_persistent_forward);
}
