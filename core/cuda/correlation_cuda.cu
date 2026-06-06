#include "correlation_cuda.h"
#include "cuda_common.h"

__global__ void count_features_kernel( // count for all feature points
    const double* __restrict__ feature_map,
    int rows, int cols,
    int half_wsize,
    int pixel_offset,
    int* __restrict__ out_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + pixel_offset;
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;

    if (row < half_wsize || row >= rows - half_wsize ||
        col < half_wsize || col >= cols - half_wsize)
        return;
    if (feature_map[idx] > 0.0)
    {
        atomicAdd(out_count, 1);
    }
}

__global__ void extract_features_kernel( // get coord for all feature points
    const double* __restrict__ feature_map,
    int rows, int cols,
    int half_wsize,
    int pixel_offset,
    int2* __restrict__ out_pts,
    int* __restrict__ out_count
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x + pixel_offset;
    int total = rows * cols;
    if (idx >= total)
        return;

    int row = idx / cols;
    int col = idx % cols;
    if (row < half_wsize || row >= rows - half_wsize ||
        col < half_wsize || col >= cols - half_wsize)
        return;
    
    if (feature_map[idx] > 0.0)
    {
        int pos = atomicAdd(out_count, 1);
        out_pts[pos] = make_int2(col, row);
    }
}

// each block for an NCC searching
// each thread for different searching objects
__global__ void ncc_match_kernel(
    const unsigned char* __restrict__ left_img,
    const unsigned char* __restrict__ right_img,
    int img_w, int img_h,
    const int2* __restrict__ feature_pts,  // (x, y) coordinates
    int num_features,
    int wsize, // window size
    int search_range,
    float ncc_threshold, // needs to be close to 1
    int2* __restrict__ out_left,
    int2* __restrict__ out_right,
    int* __restrict__ out_valid
)
{
    int idx = blockIdx.x;
    if (idx >= num_features)
       return;

    int fx = feature_pts[idx].x;
    int fy = feature_pts[idx].y;
    int half = wsize / 2;
    if (fx - half < 0 || fx + half >= img_w ||
        fy - half < 0 || fy + half >= img_h) {
        out_valid[idx] = 0;
        return;
    }

    // means for left window
    float meanL = 0.f;
    int N = wsize * wsize;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            meanL += left_img[(fy + dy) * img_w + (fx + dx)];
        }
    }
    meanL /= N;

    // sigma for left window
    float varL = 0.f;
    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            float d = left_img[(fy + dy) * img_w + (fx + dx)] - meanL;
            varL += d * d;
        }
    }
    float sigmaL = sqrtf(varL);

    int search_start_x = max(fx - search_range, half);
    int search_end_x   = min(fx + search_range, img_w - half - 1);

    if (search_start_x >= search_end_x) {
        out_valid[idx] = 0;
        return;
    }

    int best_x = -1;
    float best_ncc = -1.f;
    for (int sx = search_start_x + threadIdx.x; sx < search_end_x; sx += blockDim.x)
    {
        float meanR = 0.f;
        for (int dy = -half; dy <= half; dy++)
        {
            for (int dx = -half; dx <= half; dx++)
            {
                meanR += right_img[(fy + dy) * img_w + (sx + dx)];
            }
        }
        meanR /= N;
        float numer = 0.f, varR = 0.f;
        for (int dy = -half; dy <= half; dy++)
        {
            for (int dx = -half; dx <= half; dx++)
            {
                float dL = left_img[(fy + dy) * img_w + (fx + dx)] - meanL;
                float dR = right_img[(fy + dy) * img_w + (sx + dx)] - meanR;
                numer += dL * dR;
                varR  += dR * dR;
            }
        }
        float sigmaR = sqrtf(varR);
        float denom = sigmaL * sigmaR;
        float ncc = (denom > 1e-8f) ? numer / denom : 0.f;

        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_x   = sx;
        }
    }

    __shared__ float shared_ncc[256];
    __shared__ int   shared_x[256];
    int tid = threadIdx.x;
    shared_ncc[tid] = best_ncc;
    shared_x[tid]   = best_x;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride && shared_ncc[tid + stride] > shared_ncc[tid])
        {
            shared_ncc[tid] = shared_ncc[tid + stride];
            shared_x[tid]   = shared_x[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        if (shared_ncc[0] > ncc_threshold)
        {
            out_left[idx]   = make_int2(fx, fy);
            out_right[idx]  = make_int2(shared_x[0], fy);
            out_valid[idx]  = 1;
        } else
        {
            out_valid[idx] = 0;
        }
    }
}

std::vector<MatchPoint> correlation_match_gpu(
    const unsigned char* left_img,  int left_w,  int left_h,
    const unsigned char* right_img, int right_w, int right_h,
    const double* feature_map, int fm_rows, int fm_cols,
    int window_size, int search_range, float ncc_threshold
)
{
    select_best_cuda_device();
    int half = window_size / 2;
    int fm_total = fm_rows * fm_cols;
    int block = 256;

    // get images and feature map
    unsigned char* d_left = (unsigned char*)safe_cuda_malloc(left_w * left_h, 0.05f);
    unsigned char* d_right = (unsigned char*)safe_cuda_malloc(right_w * right_h, 0.05f);
    double* d_feature_map = (double*)safe_cuda_malloc(fm_total * sizeof(double), 0.05f);
    int* d_count = (int*)safe_cuda_malloc(sizeof(int), 0.05f);

    CUDA_CHECK(cudaMemcpy(d_left, left_img, left_w * left_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, right_img, right_w * right_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_feature_map, feature_map, fm_total * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    int pixel_batch = max_grid_blocks * block;

    // count feature points
    for (int offset = 0; offset < fm_total; offset += pixel_batch)
    {
        int N = std::min(pixel_batch, fm_total - offset);
        int grid_n = (N + block - 1) / block;
        count_features_kernel<<<grid_n, block>>>
        (
            d_feature_map, fm_rows, fm_cols, half, offset, d_count
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    int feature_count;
    CUDA_CHECK(cudaMemcpy(&feature_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (feature_count == 0)
    {
        cudaFree(d_left);  cudaFree(d_right);  cudaFree(d_feature_map);  cudaFree(d_count);
        return {};
    }

    int2* d_feature_pts = (int2*)safe_cuda_malloc(feature_count * sizeof(int2), 0.05f);
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
    for (int offset = 0; offset < fm_total; offset += pixel_batch)
    {
        int N = std::min(pixel_batch, fm_total - offset);
        int grid_n = (N + block - 1) / block;
        extract_features_kernel<<<grid_n, block>>>(
            d_feature_map, fm_rows, fm_cols, half, offset, d_feature_pts, d_count
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_feature_map);
    cudaFree(d_count);

    // matching by ncc
    int batch_size = max_grid_blocks;
    if (batch_size > feature_count)
        batch_size = feature_count;
        
    const int block_dim = 256;
    int shared_bytes = block_dim * 2 * sizeof(int);

    int2* d_left_out = (int2*)safe_cuda_malloc(feature_count * sizeof(int2), 0.05f);
    int2* d_right_out = (int2*)safe_cuda_malloc(feature_count * sizeof(int2), 0.05f);
    int* d_valid_out = (int*)safe_cuda_malloc(feature_count * sizeof(int), 0.05f);
    
    for (int offset = 0; offset < feature_count; offset += batch_size)
    {
        int N = std::min(batch_size, feature_count - offset);
        ncc_match_kernel<<<N, block_dim, shared_bytes>>>(
            d_left, d_right, left_w, left_h,
            d_feature_pts + offset, N,
            window_size, search_range, ncc_threshold,
            d_left_out  + offset,
            d_right_out + offset,
            d_valid_out + offset
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int2> h_left(feature_count), h_right(feature_count);
    std::vector<int> h_valid(feature_count);
    CUDA_CHECK(cudaMemcpy(h_left.data(),  d_left_out,  feature_count * sizeof(int2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_right.data(), d_right_out, feature_count * sizeof(int2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid.data(), d_valid_out, feature_count * sizeof(int),  cudaMemcpyDeviceToHost));

    std::vector<MatchPoint> results;
    results.reserve(feature_count);
    for (int i = 0; i < feature_count; i++)
    {
        if (h_valid[i])
        {
            results.push_back({h_left[i].x, h_left[i].y, h_right[i].x, h_right[i].y});
        }
    }

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_feature_pts);
    cudaFree(d_left_out);
    cudaFree(d_right_out);
    cudaFree(d_valid_out);
    return results;
}