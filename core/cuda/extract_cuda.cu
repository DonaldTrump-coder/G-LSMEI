#include "extract_cuda.h"
#include "cuda_common.h"

__global__ void moravec_kernel(
    const unsigned char* __restrict__ img, int w, int h, int half,
    double* __restrict__ out, double threshold
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + half;
    int row = blockIdx.y * blockDim.y + threadIdx.y + half;
    if (row >= h - half || col >= w - half) return;
    double v[4] = {0, 0, 0, 0};
    for (int k = -half; k < half; k++) {
        int idx_c = row * w + col;
        v[0] += __powf(img[(row + k) * w + col] - img[(row + k + 1) * w + col], 2);
        v[1] += __powf(img[row * w + (col + k)] - img[row * w + (col + k + 1)], 2);
        v[2] += __powf(img[(row + k) * w + (col + k)] - img[(row + k + 1) * w + (col + k + 1)], 2);
        v[3] += __powf(img[(row + k) * w + (col - k)] - img[(row + k + 1) * w + (col - k - 1)], 2);
    }
    double m = fmin(fmin(v[0], v[1]), fmin(v[2], v[3]));
    out[row * w + col] = (m > threshold) ? m : 0.0;
}

__global__ void harris_kernel(
    const unsigned char* __restrict__ img, int w, int h, int half,
    double* __restrict__ out, double threshold, double k
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + half;
    int row = blockIdx.y * blockDim.y + threadIdx.y + half;
    if (row >= h - half || col >= w - half) return;

    float gx = 0, gy = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            float p = img[(row + dy) * w + (col + dx)];
            float sx = (dx == -1) ? -1.0f : (dx == 1) ? 1.0f : 0.0f;
            if (dy == 0) sx *= 2.0f;
            gx += p * sx;
            float sy = (dy == -1) ? 1.0f : (dy == 1) ? -1.0f : 0.0f;
            if (dx == 0) sy *= 2.0f;
            gy += p * sy;
        }

    float a = 0, b = 0, c = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int r = row + dy, cl = col + dx;
            float igx = 0, igy = 0;

            igx = (float)(img[r * w + min(cl + 1, w - 1)] - img[r * w + max(cl - 1, 0)]) * 0.5f;
            igy = (float)(img[min(r + 1, h - 1) * w + cl] - img[max(r - 1, 0) * w + cl]) * 0.5f;
            a += igx * igx;
            b += igy * igy;
            c += igx * igy;
        }
    double detM = a * b - c * c;
    double traceM = a + b;
    double R = detM - k * traceM * traceM;
    out[row * w + col] = (R > threshold) ? R : 0.0;
}

__global__ void susan_kernel(
    const unsigned char* __restrict__ img, int w, int h, int R,
    double* __restrict__ out, int grayscale, float threshold
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + R;
    int row = blockIdx.y * blockDim.y + threadIdx.y + R;
    if (row >= h - R || col >= w - R) return;
    unsigned char center = img[row * w + col];
    int n = 0;

    for (int dy = -R; dy <= R; dy++)
        for (int dx = -R; dx <= R; dx++) {
            if (dx * dx + dy * dy > R * R) continue;
            if (abs((int)img[(row + dy) * w + (col + dx)] - (int)center) < grayscale)
                n++;
        }
    double response = (37.0 - n) / 37.0; // PIXEL=37
    out[row * w + col] = (response > threshold) ? response : 0.0;
}

double* extract_features_gpu(
    const unsigned char* img, int w, int h,
    char method, int* out_rows, int* out_cols
)
{
    *out_rows = h;
    *out_cols = w;
    double *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, w * h * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_out, 0, w * h * sizeof(double)));

    unsigned char *d_img;
    CUDA_CHECK(cudaMalloc(&d_img, w * h));
    CUDA_CHECK(cudaMemcpy(d_img, img, w * h, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);
    switch (method) {
        case 'M':
            moravec_kernel<<<grid, block>>>(d_img, w, h, 1, d_out, 3000.0);
            break;
        case 'H':
            harris_kernel<<<grid, block>>>(d_img, w, h, 1, d_out, 1e10, 0.04);
            break;
        case 'S':
            susan_kernel<<<grid, block>>>(d_img, w, h, 3, d_out, 20, 0.9f);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_img);
    return d_out;
}