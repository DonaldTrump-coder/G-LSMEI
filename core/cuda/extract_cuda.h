#pragma once

double* extract_features_gpu(
    const unsigned char* img, int w, int h,
    char method, // 'M'=Moravec, 'H'=Harris, 'S'=SUSAN
    int* out_rows, int* out_cols
);