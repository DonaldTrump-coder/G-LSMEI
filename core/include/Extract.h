#pragma once
//Jiahao Zhou
//jeiluosmith@gmail.com
//git@github:Jeiluo

#include <iostream>
#include <math.h>
#include "opencv2/opencv.hpp"
using namespace cv;

class PointFeature {
private:
	cv::Mat Img;
public:
	const cv::Mat& getOrigImg() const { return Img; }

	cv::Mat Moravec_calculate(cv::Mat&);
	cv::Mat Harris_calculate(cv::Mat&);
	cv::Mat SUSAN_calculate(cv::Mat&);
    #ifdef HAS_CUDA
    static double* extract_gpu(const unsigned char* img, int w, int h, char method, int* rows, int* cols);
    static void free_gpu_feature_map(double* d_ptr);
    #endif
};