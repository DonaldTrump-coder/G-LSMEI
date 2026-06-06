//Jiahao Zhou
//jeiluosmith@gmail.com
//git@github:Jeiluo

#include "Extract.h"
#include <array>

const int WINDOWSIZE = 3;

const int THRESHOLD_Moravec = 3000;

const long THRESHOLD_Harris = 1e10;
const double K = 0.04;

const int R = 3; // �뾶
const int GRAYSCALE = 20; // �ҶȲ���ֵ
const int PIXEL = 37; // ��ģ��������
const float THRESHOLD_SUSAN = 0.9; // ��Ӧֵ��ֵ

Mat PointFeature::Moravec_calculate(Mat& img) {
	Mat MoravecVar = Mat::zeros(img.rows, img.cols, CV_64FC1);

	for (int i = WINDOWSIZE / 2; i < img.rows - WINDOWSIZE / 2; i++)
		for (int j = WINDOWSIZE / 2; j < img.cols - WINDOWSIZE / 2; j++) {
			std::array<double, 4>Var{ 0,0,0,0 };
			for (int k = -WINDOWSIZE / 2; k < WINDOWSIZE / 2; k++) {
				if ((!img.at<uchar>(i + k, j)) ||
					(!img.at<uchar>(i + k + 1, j))
					|| (!img.at<uchar>(i, j + k))
					|| (!img.at<uchar>(i, j + k + 1))
					|| (!img.at<uchar>(i + k, j + k))
					|| (!img.at<uchar>(i + k + 1, j + k + 1))
					|| (!img.at<uchar>(i + k, j - k))
					|| (!img.at<uchar>(i + k + 1, j - k - 1)))
					continue;
				Var[0] += pow((img.at<uchar>(i + k, j) - img.at<uchar>(i + k + 1, j)), 2);//��ֱ
				Var[1] += pow((img.at<uchar>(i, j + k) - img.at<uchar>(i, j + k + 1)), 2);//ˮƽ
				Var[2] += pow((img.at<uchar>(i + k, j + k) - img.at<uchar>(i + k + 1, j + k + 1)), 2);//���Խ���
				Var[3] += pow((img.at<uchar>(i + k, j - k) - img.at<uchar>(i + k + 1, j - k - 1)), 2);//���Խ���
			}
			double variant = *std::min_element(Var.begin(), Var.end());

			if (variant > THRESHOLD_Moravec)
				MoravecVar.at<double>(i, j) = variant;
		}
	return MoravecVar;
}

cv::Mat PointFeature::Harris_calculate(cv::Mat& img) {
	Mat HarrisVar = Mat::zeros(img.rows, img.cols, CV_64FC1);

	std::array<Mat, 2>I
	{ Mat::zeros(img.rows,img.cols,CV_64FC1),Mat::zeros(img.rows,img.cols,CV_64FC1) };
	/*
	Sobel(img, I[0], CV_64FC1, 1, 0, 3);
	Sobel(img, I[1], CV_64FC1, 0, 1, 3);
	*///Sobel���ӿ��Լ����ݶ�

	std::array<Mat, 3>Gs
	{ Mat::zeros(img.rows,img.cols,CV_64FC1) ,Mat::zeros(img.rows,img.cols,CV_64FC1) ,Mat::zeros(img.rows,img.cols,CV_64FC1) };
	double Tx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
	double Ty[3][3] = { {1,2,1},{0,0,0},{-1,-2,-1} };
	for (int i = WINDOWSIZE / 2; i < img.rows - WINDOWSIZE / 2; i++)
		for (int j = WINDOWSIZE / 2; j < img.cols - WINDOWSIZE / 2; j++) {
			double gx = 0, gy = 0;
			for (int m = -WINDOWSIZE / 2; m <= WINDOWSIZE / 2; m++)
				for (int n = -WINDOWSIZE / 2; n <= WINDOWSIZE / 2; n++)
					if ((i + m) > 0 && (i + m) < img.rows && (j + n) > 0 && (j + n) < img.cols) {
						gx += img.at<uchar>(i + m, j + n) * Tx[m + WINDOWSIZE / 2][n + WINDOWSIZE / 2];
						gy += img.at<uchar>(i + m, j + n) * Ty[m + WINDOWSIZE / 2][n + WINDOWSIZE / 2];
					}
			I[0].at<double>(i, j) = gx;
			I[1].at<double>(i, j) = gy;
		}

	GaussianBlur(I[0].mul(I[0]), Gs[0], Size(3, 3), 0);
	GaussianBlur(I[1].mul(I[1]), Gs[1], Size(3, 3), 0);
	GaussianBlur(I[0].mul(I[1]), Gs[2], Size(3, 3), 0);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			double detM =
				Gs[0].at<double>(i, j) * Gs[1].at<double>(i, j) -
				Gs[2].at<double>(i, j) * Gs[2].at<double>(i, j);
			double traceM =
				Gs[0].at<double>(i, j) + Gs[1].at<double>(i, j);
			double R = detM - K * traceM * traceM;
			if (R > THRESHOLD_Harris)
				HarrisVar.at<double>(i, j) = R;
		}
	return HarrisVar;
}

cv::Mat PointFeature::SUSAN_calculate(cv::Mat& img) {
	Mat SUSANVar = Mat::zeros(img.rows, img.cols, CV_64FC1);
	std::vector<cv::Point> mask;
	for (int y = -R; y <= R; y++) {
		for (int x = -R; x <= R; x++) {
			if (x * x + y * y <= R * R) {
				mask.push_back(cv::Point(x, y));
			}
		}
	}//����Բ����ģ

	for (int i = R; i < img.rows - R; i++) {
		for (int j = R; j < img.cols - R; j++) {
			int n = 0;//��ģ�����������ػҶȲ�С����ֵ��������
			for (const cv::Point& offset : mask)
				if (std::abs(img.at<uchar>(i, j) -
					img.at<uchar>(i + offset.y, j + offset.x))
					< GRAYSCALE) {
					n++;
				}
			double response = static_cast<double>(PIXEL - n) / PIXEL;
			if (response > THRESHOLD_SUSAN) { // �����Ƶ�������������ֵ����Ϊ�ǽǵ�
				SUSANVar.at<double>(i, j) = response;
			}
		}
	}
	return SUSANVar;
}

#ifdef HAS_CUDA
#include "cuda_common.h"
#include "extract_cuda.h"

double* PointFeature::extract_gpu(
    const unsigned char* img, int w, int h,
    char method, int* rows, int* cols
)
{
    return extract_features_gpu(img, w, h, method, rows, cols);
}
#endif