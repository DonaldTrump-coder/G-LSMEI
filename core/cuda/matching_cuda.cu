#include "matching_cuda.h"
#include "cuda_common.h"

__device__ float sample(float x, float y,
                        const unsigned char* __restrict__ img,
                        int w, int h)
{
    x = fminf(fmaxf(x, 0.f), (float)(w - 1));
    y = fminf(fmaxf(y, 0.f), (float)(h - 1));
    int x0 = (int)x, y0 = (int)y;
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float dx = x - x0, dy = y - y0;
    float i00 = img[y0 * w + x0], i10 = img[y0 * w + x1];
    float i01 = img[y1 * w + x0], i11 = img[y1 * w + x1];
    return i00 * (1.f - dx) * (1.f - dy) +
           i10 * dx * (1.f - dy) +
           i01 * (1.f - dx) * dy +
           i11 * dx * dy;
}

__device__ bool invert_8x8(float A[8][8])
{
    int n = 8;
    float aug[8][16];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
        for (int j = n; j < 2*n; j++) aug[i][j] = (i == j - n) ? 1.f : 0.f;
    }
    for (int col = 0; col < n; col++)
    {
        int pivot = col;
        float max_val = fabsf(aug[col][col]);
        for (int row = col + 1; row < n; row++)
        {
            if (fabsf(aug[row][col]) > max_val)
            {
                max_val = fabsf(aug[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-12f) return false;
        if (pivot != col)
        {
            for (int j = 0; j < 2*n; j++)
            {
                float tmp = aug[col][j];
                aug[col][j] = aug[pivot][j];
                aug[pivot][j] = tmp;
            }
        }
        float diag = aug[col][col];
        for (int j = 0; j < 2*n; j++) aug[col][j] /= diag;
        for (int row = 0; row < n; row++) {
            if (row != col)
            {
                float factor = aug[row][col];
                for (int j = 0; j < 2*n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = aug[i][j + n];
    return true;
}

__device__ void mat_mul_8x8_8x1(const float A[8][8], const float B[8], float C[8])
{
    for (int i = 0; i < 8; i++)
    {
        C[i] = 0.f;
        for (int j = 0; j < 8; j++) C[i] += A[i][j] * B[j];
    }
}

__global__ void lsm_adjust_kernel(
    const unsigned char* __restrict__ left_img,
    const unsigned char* __restrict__ right_img,
    int img_w, int img_h,
    int right_w, int right_h,
    const int* __restrict__ left_cx,
    const int* __restrict__ left_cy,
    const int* __restrict__ right_cx,
    const int* __restrict__ right_cy,
    int num_points, int wsize,
    float d_corr_thresh, int max_iter,
    float* __restrict__ results
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points)
        return;

    // guard
    int half, inner_N, k_inner;
    if (wsize < 5)
    {
        int off = idx * 18;
        for (int k = 0; k < 18; k++) results[off + k] = 0.f;
        return;
    }

    int ws = wsize;
    if (ws > WIND_SIZE)
        ws = WIND_SIZE;
    if (ws % 2 == 0)
        ws--;
    half = ws / 2;
    inner_N = ws - 2;
    k_inner = inner_N / 2;
    
    int lx = left_cx[idx], ly = left_cy[idx];
    int rx = right_cx[idx], ry = right_cy[idx];

    if (lx - half < 0 || lx + half >= img_w ||
        ly - half < 0 || ly + half >= img_h ||
        rx - half < 0 || rx + half >= right_w ||
        ry - half < 0 || ry + half >= right_h) {
        int off = idx * 18;
        for (int k = 0; k < 18; k++) results[off + k] = 0.f;
        results[off + 8]  = NAN;
        results[off + 9]  = NAN;
        return;
    }

    float left_win[WIND_SIZE][WIND_SIZE];
    for (int dy = -half; dy <= half; dy++)
        for (int dx = -half; dx <= half; dx++)
            left_win[dy + half][dx + half] =
                left_img[(ly + dy) * img_w + (lx + dx)];

    float h0 = 0.f, h1 = 1.f;
    float a0 = 0.f, a1 = 1.f, a2 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 1.f;

    float corr = 0.f;
    bool first = true;
    bool stop = false;
    int iter = 0;

    while (!stop && iter < max_iter)
    {
        float right_win[WIND_SIZE][WIND_SIZE], g2[WIND_SIZE][WIND_SIZE];
        for (int dy = -half; dy <= half; dy++)
        {
            for (int dx = -half; dx <= half; dx++)
            {
                float x2 = a0 + a1 * (rx + dx) + a2 * (ry + dy);
                float y2 = b0 + b1 * (rx + dx) + b2 * (ry + dy);
                float val = sample(x2, y2, right_img, right_w, right_h);
                val = h0 + h1 * val;
                val = fminf(fmaxf(val, 0.f), 255.f);
                right_win[dy + half][dx + half] = val;
                g2[dy + half][dx + half] = val;
            }
        }

        float meanL = 0.f, meanR = 0.f;
        int N = ws * ws;
        for (int dy = 0; dy < ws; dy++)
            for (int dx = 0; dx < ws; dx++)
            {
                meanL += left_win[dy][dx];
                meanR += right_win[dy][dx];
            }
        meanL /= N; meanR /= N;
        float L_R = 0.f, L_var = 0.f, R_var = 0.f;
        for (int dy = 0; dy < ws; dy++)
            for (int dx = 0; dx < ws; dx++) {
                float dL = left_win[dy][dx] - meanL;
                float dR = right_win[dy][dx] - meanR;
                L_R += dL * dR;
                L_var += dL * dL;
                R_var += dR * dR;
            }
        float new_corr = L_R / sqrtf(L_var * R_var + 1e-10f);
        if (first)
        {
            corr = new_corr; first = false;
        }
        else
        {
            if (fabsf(new_corr - corr) < d_corr_thresh) stop = true;
            corr = new_corr;
        }

        float g2_dx[WIND_SIZE - 2][WIND_SIZE - 2], g2_dy[WIND_SIZE - 2][WIND_SIZE - 2];
        for (int i = 0; i < inner_N; i++)
            for (int j = 0; j < inner_N; j++)
            {
                g2_dx[j][i] = 0.5f * (g2[j+1][i+2] - g2[j+1][i]);
                g2_dy[j][i] = 0.5f * (g2[j+2][i+1] - g2[j][i+1]);
            }

        float BTB[8][8] = {{0}};
        float BTL[8] = {0};
        for (int i = 0; i < inner_N; i++)
        {
            for (int j = 0; j < inner_N; j++)
            {
                int x_p = rx - k_inner + i;
                int y_p = ry - k_inner + j;
                float row[8];
                row[0] = 1.f;
                row[1] = g2[j+1][i+1];
                row[2] = h1 * g2_dx[j][i];
                row[3] = h1 * x_p * g2_dx[j][i];
                row[4] = h1 * y_p * g2_dx[j][i];
                row[5] = h1 * g2_dy[j][i];
                row[6] = h1 * x_p * g2_dy[j][i];
                row[7] = h1 * y_p * g2_dy[j][i];
                float dl = left_win[j+1][i+1] - g2[j+1][i+1];
                for (int r = 0; r < 8; r++)
                {
                    BTL[r] += row[r] * dl;
                    for (int c = 0; c < 8; c++)
                        BTB[r][c] += row[r] * row[c];
                }
            }
        }

        if (!invert_8x8(BTB)) break;
        float dx_vec[8];
        mat_mul_8x8_8x1(BTB, BTL, dx_vec);

        float dh0 = dx_vec[0], dh1 = dx_vec[1];
        float da0 = dx_vec[2], da1 = dx_vec[3], da2 = dx_vec[4];
        float db0 = dx_vec[5], db1 = dx_vec[6], db2 = dx_vec[7];
        h0 = h0 + dh0 + h0 * dh1;
        h1 = h1 + h1 * dh1;
        a0 = a0 + da0 + a0 * da1 + b0 * da2;
        a1 = a1 + a1 * da1 + b1 * da2;
        a2 = a2 + a2 * da1 + b2 * da2;
        b0 = b0 + db0 + a0 * db1 + b0 * db2;
        b1 = b1 + a1 * db1 + b1 * db2;
        b2 = b2 + a2 * db1 + b2 * db2;
        iter++;
    }

    float matched_x = a0 + rx * a1 + ry * a2;
    float matched_y = b0 + rx * b1 + ry * b2;

    int offset = idx * 18;
    if (matched_x < 0.f || matched_x >= (float)right_w ||
        matched_y < 0.f || matched_y >= (float)right_h ||
        isnan(matched_x) || isnan(matched_y))
    {
        results[offset + 8] = NAN;
        results[offset + 9] = NAN;
        return;
    }
    results[offset + 0]  = h0;
    results[offset + 1]  = h1;
    results[offset + 2]  = a0;
    results[offset + 3]  = a1;
    results[offset + 4]  = a2;
    results[offset + 5]  = b0;
    results[offset + 6]  = b1;
    results[offset + 7]  = b2;
    results[offset + 8]  = matched_x;
    results[offset + 9]  = matched_y;
    results[offset + 10] = 0.f;
    results[offset + 11] = 0.f;
    results[offset + 12] = corr;
    results[offset + 13] = 0.f;
    results[offset + 14] = 0.f;
    results[offset + 15] = 0.f;
    results[offset + 16] = (float)iter;
    results[offset + 17] = stop ? 1.f : 0.f;
}

std::vector<AdjustResult> matching_adjust_gpu(
    const unsigned char* left_img,  int img_w,  int img_h,
    const unsigned char* right_img, int right_w, int right_h,
    const std::vector<std::pair<int,int>>& left_centers,
    const std::vector<std::pair<int,int>>& right_centers,
    int window_size, float d_corr_threshold, int max_iterations
)
{
    select_best_cuda_device();

    int N = (int)left_centers.size();
    if (N == 0)
        return {};
    
    unsigned char* d_left  = (unsigned char*)safe_cuda_malloc(img_w * img_h, 0.05f);
    unsigned char* d_right = (unsigned char*)safe_cuda_malloc(right_w * right_h, 0.05f);
    CUDA_CHECK(cudaMemcpy(d_left, left_img, img_w * img_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_right, right_img, right_w * right_h, cudaMemcpyHostToDevice));

    std::vector<int> h_lx(N), h_ly(N), h_rx(N), h_ry(N);
    for (int i = 0; i < N; i++)
    {
        h_lx[i] = left_centers[i].first;
        h_ly[i] = left_centers[i].second;
        h_rx[i] = right_centers[i].first;
        h_ry[i] = right_centers[i].second;
    }
    int *d_lx = (int*)safe_cuda_malloc(N * sizeof(int), 0.05f);
    int *d_ly = (int*)safe_cuda_malloc(N * sizeof(int), 0.05f);
    int *d_rx = (int*)safe_cuda_malloc(N * sizeof(int), 0.05f);
    int *d_ry = (int*)safe_cuda_malloc(N * sizeof(int), 0.05f);

    CUDA_CHECK(cudaMemcpy(d_lx, h_lx.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ly, h_ly.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rx, h_rx.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ry, h_ry.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    float* d_results = (float*)safe_cuda_malloc(N * 18 * sizeof(float), 0.10f);

    const int threads = 128;
    const int batch_size = max_grid_blocks * threads;
    int bs = batch_size;
    if (bs > N) bs = N;

    for (int offset = 0; offset < N; offset += bs)
    {
        int M = std::min(bs, N - offset);
        int blocks = (M + threads - 1) / threads;
        lsm_adjust_kernel<<<blocks, threads>>>(
            d_left, d_right, img_w, img_h, right_w, right_h,
            d_lx + offset, d_ly + offset,
            d_rx + offset, d_ry + offset,
            M,
            window_size, d_corr_threshold, max_iterations,
            d_results + offset * 18
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_results(N * 18);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, N * 18 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<AdjustResult> out(N);
    for (int i = 0; i < N; i++)
    {
        int off = i * 18;
        out[i].h0 = h_results[off + 0]; 
        out[i].h1 = h_results[off + 1];
        out[i].a0 = h_results[off + 2]; 
        out[i].a1 = h_results[off + 3];
        out[i].a2 = h_results[off + 4]; 
        out[i].b0 = h_results[off + 5];
        out[i].b1 = h_results[off + 6]; 
        out[i].b2 = h_results[off + 7];
        out[i].matched_x = h_results[off + 8];
        out[i].matched_y = h_results[off + 9];
        out[i].delta0 = h_results[off + 10];
        out[i].SNR = h_results[off + 11];
        out[i].rho = h_results[off + 12];
        out[i].deltag = h_results[off + 13];
        out[i].deltag_ = h_results[off + 14];
        out[i].deltax = h_results[off + 15];
        out[i].iterations = (int)h_results[off + 16];
        out[i].converged = h_results[off + 17] > 0.5f;
    }
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_lx);
    cudaFree(d_ly);
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_results);
    return out;
}

int get_gpu_device_count()
{
    return get_cuda_device_count();
}