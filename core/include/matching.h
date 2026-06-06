#include "opencv2/opencv.hpp"
#include <string>
#include "matrix.h"
#include "tools.h"

class matching
{
    private:
    cv::Mat left_img;
    cv::Mat right_img;
    int window_size = 15;
    cv::Mat left_window;
    cv::Mat right_window;
    cv::Mat left_window_centered;
    int W,H;
    int leftx, lefty, rightx, righty;
    double new_rightx, new_righty;
    Matrix X;
    Matrix B;
    Matrix L;
    Matrix x;
    Matrix v;

    cv::Mat g2;
    cv::Mat g2_dx, g2_dy;
    cv::Mat distortion_window;

    double corr;
    double d_corr;
    double d_corr_thres = 0.04;
    bool first = 1;
    bool stop = 0;
    double delta0, deltag, deltag_, deltax;
    double SNR;
    double rho;

    double min_x_left;
    double min_y_left;
    double min_x_right;
    double min_y_right;

    int matching_window_size;
    double ncc_threshold;

    public:
    matching(std::string left_path, std::string right_path);
    void set_params(int window_size, double threshold);
    void set_matching_params(int matching_window_size, double ncc_threshold);
    void set_centers(int left_x, int left_y, int right_x, int right_y); //construct windows
    void disp_windows();
    void params_init();
    void centerize();
    void get_g2(); //right window 重采样
    float sample_img(cv::Mat img, double x, double y);
    void radioCorrection();
    void get_dg(); //对g2求导
    void construct_matrices(); //构造系数矩阵
    void adjustment();
    void update();
    void get_corr();
    void calculate();
    void precision();
    cv::Mat get_left_window();
    cv::Mat get_right_window();
    void get_result();
    void get_matched_points(std::string savepath);
    double get_matched_y();
    double get_matched_x();
    double get_delta0();
    double get_SNR();
    double get_rho();
    double get_deltag();
    double get_deltag_();
    double get_deltax();
    double get_h0();
    double get_h1();
    double get_a0();
    double get_a1();
    double get_a2();
    double get_b0();
    double get_b1();
    double get_b2();

    std::vector<std::tuple<int, int, double, double>>
    batch_adjust_gpu(
        const std::vector<std::pair<int, int>>& left_centers,
        const std::vector<std::pair<int, int>>& right_centers,
        int window_size, double d_corr_threshold,
        int max_iterations = 20
    );
    static int gpu_device_count();
};