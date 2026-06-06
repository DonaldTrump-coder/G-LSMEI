#pragma once
//Jiahao Zhou
//jeiluosmith@gmail.com
//git@github:Jeiluo

#include <iostream>	
#include <math.h>
#include <vector>
#include "opencv2/opencv.hpp"
using namespace cv;

class CorrelationMatch {
private:
	std::vector<Point>LeftSame;
	std::vector<Point>RigSame;
public:
	std::vector<Point> getLeftSame() const { return LeftSame; }
	std::vector<Point> getRigSame() const { return RigSame; }

	cv::Mat matOperator(char op, cv::Mat&);
	void saveResult(const std::string& savepath);
	void Calculate(cv::Mat& LefImg, cv::Mat& RigImg, int WINDOWSIZE, double NCC_THRESHOLD);
    #ifdef HAS_CUDA
    void CalculateGPU(cv::Mat& LefImg, cv::Mat& RigImg, int WINDOWSIZE, double NCC_THRESHOLD);
    #endif
};
