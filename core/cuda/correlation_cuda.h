#pragma once

#include <vector>
#include <utility>

struct MatchPoint {
    int left_x, left_y;
    int right_x, right_y;
};

std::vector<MatchPoint> correlation_match_gpu(
    const unsigned char* left_img,  int left_w,  int left_h,
    const unsigned char* right_img, int right_w, int right_h,
    const double* feature_map, int fm_rows, int fm_cols,
    int window_size, int search_range, float ncc_threshold
);