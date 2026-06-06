#pragma once
#include <vector>
#include <utility>

const int WIND_SIZE = 45;

struct AdjustResult
{
    double h0, h1, a0, a1, a2, b0, b1, b2;
    double matched_x, matched_y;
    double delta0, SNR, rho, deltag, deltag_, deltax;
    int iterations;
    bool converged;
};

std::vector<AdjustResult> matching_adjust_gpu(
    const unsigned char* left_img, int img_w, int img_h,
    const unsigned char* right_img, int right_w, int right_h,
    const std::vector<std::pair<int,int>>& left_centers, // (x, y)
    const std::vector<std::pair<int,int>>& right_centers,
    int window_size, float d_corr_threshold, int max_iterations
);