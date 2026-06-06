#include "matching.h"
#include "Correlation.h"

matching::matching(std::string left_path,
                std::string right_path
                )
{
    left_img = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    right_img = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    W = left_img.cols;
    H = right_img.rows;
}

void matching::set_params(int size = 15, double threshold = 0.04)
{
    window_size = size;
    d_corr_thres = threshold;
    left_window = cv::Mat::zeros(window_size, window_size, CV_32F);
    right_window = cv::Mat::zeros(window_size, window_size, CV_32F);
    g2_dx = cv::Mat::zeros(window_size-2, window_size-2, CV_32F);
    g2_dy = cv::Mat::zeros(window_size-2, window_size-2, CV_32F); //初始化变量：左右影像窗口和左右窗口差分值
}

void matching::set_centers(int left_x, int left_y, int right_x, int right_y)
{
    int k = (int)(window_size/2);
    if (left_x - k < 0 || left_x + k >= W ||
        left_y - k < 0 || left_y + k >= H ||
        right_x - k < 0 || right_x + k >= W ||
        right_y - k < 0 || right_y + k >= H) {
        leftx = -1;
        return;
    }
    leftx = left_x;
    lefty = left_y;
    rightx = right_x;
    righty = right_y; //record the centers
    for(int i = -k; i<k+1; i++) //x
    {
        for(int j = -k; j<k+1;j++) //y
        {
            left_window.at<float>(j+k, i+k) = left_img.at<uchar>(left_y+j, left_x+i);
            right_window.at<float>(j+k, i+k) = right_img.at<uchar>(right_y+j, right_x+i);
        }
    }
}

void matching::disp_windows() //展示影像窗口
{
    cv::Mat left_win_show;
    cv::Mat right_win_show;
    cv::normalize(left_window, left_win_show, 0, 1, cv::NORM_MINMAX);
    cv::normalize(right_window, right_win_show, 0, 1, cv::NORM_MINMAX); //影像窗口用了32位浮点数存储，因此在显示时要归一化至0-1之间再显示
    //cv::imshow("Left",left_win_show);
    cv::imshow("Right",right_win_show); //展示图像
    cv::waitKey(0);
}

void matching::params_init()
{
    X = Matrix(8,1); //h0,h1,a0,a1,a2,b0,b1,b2
    X.SetMatrix_ele(0,0,0); //h0
    X.SetMatrix_ele(1,0,1); //h1
    X.SetMatrix_ele(2,0,0); //a0
    X.SetMatrix_ele(3,0,1); //a1
    X.SetMatrix_ele(4,0,0); //a2
    X.SetMatrix_ele(5,0,0); //b0
    X.SetMatrix_ele(6,0,0); //b1
    X.SetMatrix_ele(7,0,1); //b2

    B = Matrix((window_size-2)*(window_size-2), 8);
    L = Matrix((window_size-2)*(window_size-2), 1);
}

float matching::sample_img(cv::Mat img, double x, double y)
{
    int width = img.cols;
    int height = img.rows;

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > width - 1) 
    {
        x = width - 1;
    }
    if (y > height - 1) 
    {
        y = height - 1;
    }

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, width - 1);
    int y1 = std::min(y0 + 1, height - 1);

    double dx = x - x0;
    double dy = y - y0;

    double I00 = img.at<uchar>(y0, x0);
    double I10 = img.at<uchar>(y0, x1);
    double I01 = img.at<uchar>(y1, x0);
    double I11 = img.at<uchar>(y1, x1);

    double I = I00 * (1 - dx) * (1 - dy) +
               I10 * dx * (1 - dy) +
               I01 * (1 - dx) * dy +
               I11 * dx * dy;
    return static_cast<float>(clamp(I, 0.0, 255.0));
}

void matching::get_g2()
{
    double a0 = X.getMatrix_ele(2,0);
    double a1 = X.getMatrix_ele(3,0);
    double a2 = X.getMatrix_ele(4,0);
    double b0 = X.getMatrix_ele(5,0);
    double b1 = X.getMatrix_ele(6,0);
    double b2 = X.getMatrix_ele(7,0);
    int k = (int)(window_size/2);
    for(int i = -k; i<k+1; i++)
    {
        for(int j = -k; j<k+1;j++)
        {
            int x = rightx + i;
            int y = righty + j; //变形前的像素点坐标
            double x2 = a0 + a1*x + a2*y;
            double y2 = b0 + b1*x + b2*y; //变形后的像素点坐标
            right_window.at<float>(j+k, i+k) = sample_img(right_img, x2, y2);
        }
    }
    g2 = right_window.clone();
}

void matching::radioCorrection()
{
    double h0 = X.getMatrix_ele(0,0);
    double h1 = X.getMatrix_ele(1,0);

    int k = (int)(window_size/2);
    for(int i = -k; i<k+1; i++)
    {
        for(int j = -k; j<k+1;j++)
        {
            float val = right_window.at<float>(j+k, i+k);
            double corrected = h0 + h1 * val;
            right_window.at<float>(j+k, i+k) = static_cast<float>(clamp(corrected, 0.0, 255.0));
        }
    }
}

void matching::get_dg()
{
    for(int i = 0; i<window_size-2; i++)
    {
        for(int j = 0;j<window_size-2;j++)
        {
            g2_dx.at<float>(j,i) = 0.5*(g2.at<float>(j+1,i+2)-g2.at<float>(j+1,i));
            g2_dy.at<float>(j,i) = 0.5*(g2.at<float>(j+2,i+1)-g2.at<float>(j,i+1));
        }
    }
}

void matching::construct_matrices()
{
    double h1 = X.getMatrix_ele(1,0);
    int pix_counter = 0;
    int k = (int)((window_size-2)/2);
    for(int i = 0;i<window_size-2; i++) //x
    {
        for(int j = 0;j<window_size-2; j++) //y
        {
            int x = rightx - k + i;
            int y = righty - k + j;
            B.SetMatrix_ele(pix_counter, 0, 1);
            B.SetMatrix_ele(pix_counter, 1, g2.at<float>(j+1, i+1));

            B.SetMatrix_ele(pix_counter, 2, h1*g2_dx.at<float>(j,i));
            B.SetMatrix_ele(pix_counter, 3, h1*x*g2_dx.at<float>(j,i));
            B.SetMatrix_ele(pix_counter, 4, h1*y*g2_dx.at<float>(j,i));

            B.SetMatrix_ele(pix_counter, 5, h1*g2_dy.at<float>(j,i));
            B.SetMatrix_ele(pix_counter, 6, h1*x*g2_dy.at<float>(j,i));
            B.SetMatrix_ele(pix_counter, 7, h1*y*g2_dy.at<float>(j,i));

            float dl = left_window.at<float>(j+1, i+1) - right_window.at<float>(j+1, i+1);
            L.SetMatrix_ele(pix_counter, 0, dl);
            pix_counter++;
        }
    }
}

void matching::adjustment()
{
    x = (B.trans()*B).inv()*B.trans()*L;
}

void matching::update()
{
    double dh0 = x.getMatrix_ele(0,0);
    double dh1 = x.getMatrix_ele(1,0);
    double da0 = x.getMatrix_ele(2,0);
    double da1 = x.getMatrix_ele(3,0);
    double da2 = x.getMatrix_ele(4,0);
    double db0 = x.getMatrix_ele(5,0);
    double db1 = x.getMatrix_ele(6,0);
    double db2 = x.getMatrix_ele(7,0);

    double h0 = X.getMatrix_ele(0,0);
    double h1 = X.getMatrix_ele(1,0);
    double a0 = X.getMatrix_ele(2,0);
    double a1 = X.getMatrix_ele(3,0);
    double a2 = X.getMatrix_ele(4,0);
    double b0 = X.getMatrix_ele(5,0);
    double b1 = X.getMatrix_ele(6,0);
    double b2 = X.getMatrix_ele(7,0);
    
    double new_h0 = h0 + dh0 + h0*dh1;
    double new_h1 = h1 + h1*dh1;
    double new_a0 = a0 + da0 + a0*da1 + b0*da2;
    double new_a1 = a1 + a1*da1 + b1*da2;
    double new_a2 = a2 + a2*da1 + b2*da2;
    double new_b0 = b0 + db0 + a0*db1 + b0*db2;
    double new_b1 = b1 + a1*db1 + b1*db2;
    double new_b2 = b2 + a2*db1 + b2*db2;
    
    
    /*
    double new_h0 = h0 + dh0;
    double new_h1 = h1 + dh1;
    double new_a0 = a0 + da0;
    double new_a1 = a1 + da1;
    double new_a2 = a2 + da2;
    double new_b0 = b0 + db0;
    double new_b1 = b1 + db1;
    double new_b2 = b2 + db2;
    */

    X.SetMatrix_ele(0,0,new_h0);
    X.SetMatrix_ele(1,0,new_h1);
    X.SetMatrix_ele(2,0,new_a0);
    X.SetMatrix_ele(3,0,new_a1);
    X.SetMatrix_ele(4,0,new_a2);
    X.SetMatrix_ele(5,0,new_b0);
    X.SetMatrix_ele(6,0,new_b1);
    X.SetMatrix_ele(7,0,new_b2);
}

void matching::get_corr()
{
    double left_mean = 0;
    double right_mean = 0;
    int N = window_size*window_size;
    for(int i=0;i<window_size;i++)
    {
        for(int j=0;j<window_size;j++)
        {
            left_mean += left_window.at<float>(j,i);
            right_mean += right_window.at<float>(j,i);
        }
    }
    left_mean /= N;
    right_mean /= N;

    double L_R_var = 0;
    double L_var = 0;
    double R_var = 0;
    for (int i = 0; i < window_size; i++)
    {
        for (int j = 0; j < window_size; j++)
        {
            double dL = left_window.at<float>(j,i)-left_mean;
            double dR = right_window.at<float>(j,i)-right_mean;
            L_R_var += dL*dR;
            L_var += dL*dL;
            R_var += dR*dR;
        }
    }
    double result = L_R_var/std::sqrt(L_var*R_var);
    if(first == 1)
    {
        corr = result;
        first = 0;
    }
    else
    {
        d_corr = result - corr;
        corr = result;
    }
}

void matching::centerize()
{
    left_window.convertTo(left_window_centered, CV_32F);
    cv::Scalar meanVal = cv::mean(left_img);
    float meanGray = static_cast<float>(meanVal[0]);

    left_window_centered = left_window_centered - meanGray;
}

void matching::precision()
{
    double a0 = X.getMatrix_ele(2,0);
    double a1 = X.getMatrix_ele(3,0);
    double a2 = X.getMatrix_ele(4,0);
    double b0 = X.getMatrix_ele(5,0);
    double b1 = X.getMatrix_ele(6,0);
    double b2 = X.getMatrix_ele(7,0);
    double h0 = X.getMatrix_ele(0,0);
    double h1 = X.getMatrix_ele(1,0);
    v = B * x - L;
    delta0 = std::sqrt((v.trans()*v).getMatrix_ele(0,0)/((window_size-2)*(window_size-2)-8));
    deltag = 0;
    centerize();
    deltag_ = 0;
    for(int i = 0; i<window_size-2;i++)
    {
        for(int j = 0;j<window_size-2;j++)
        {
            double g = left_window_centered.at<float>(j+1,i+1);
            deltag += g*g;

            double dx = g2_dx.at<float>(j,i);
            double dy = g2_dy.at<float>(j,i);
            deltag_ += dx*dx + dy*dy;
        }
    }
    deltag = std::sqrt(deltag/((window_size-2)*(window_size-2)));
    deltag_ = std::sqrt(deltag_/((window_size-2)*(window_size-2)));
    SNR = deltag / delta0;
    rho = std::sqrt(1-1/(SNR*SNR));
    deltax = std::sqrt(((1-rho*rho)*deltag*deltag)/((window_size-2)*(window_size-2)*deltag_*deltag_));

    new_rightx = a0 + rightx*a1 + righty * a2;
    new_righty = b0 + rightx*b1 + righty * b2;
    int k = (int)((window_size-2)/2);
    int pix_counter = 0;
    double v_min = v.getMatrix_ele(0,0);
    min_x_left = leftx - k;
    min_y_left = lefty - k;
    min_x_right = rightx - k;
    min_y_right = righty - k;
    for(int i=-k;i<k+1;i++)
    {
        for(int j=-k;j<k+1;j++)
        {
            double v_present = v.getMatrix_ele(pix_counter,0);
            if(dabs(v_present)<dabs(v_min))
            {
                v_min = v_present;
                min_x_left = leftx + i;
                min_y_left = lefty + j;
                min_x_right = rightx + i;
                min_y_right = righty + j;
            }
            pix_counter++;
        }
    }
    double new_min_x_right = a0 + min_x_right*a1 + min_y_right * a2;
    double new_min_y_right = b0 + min_x_right*b1 + min_y_right * b2;
    min_x_right = new_min_x_right;
    min_y_right = new_min_y_right;
}

void matching::calculate()
{
    first = 1;
    stop = 0;
    params_init();
    int times = 0;
    while(1)
    {
        get_g2();
        radioCorrection();
        //right_window.convertTo(right_window, CV_32F, 1.0 / 255.0);
        //cv::Mat img_8u;
        //right_window.convertTo(img_8u, CV_8U, 255.0);
        //cv::imwrite(std::to_string(times) + ".jpg", img_8u);
        if(first == 1)
        {
            get_corr();
        }
        else
        {
            get_corr();
            if(dabs(d_corr) < d_corr_thres)
            {
                stop = 1;
            }
        }
        get_dg();
        construct_matrices();
        adjustment();
        update();
        times++;
        if(stop == 1 || times > 20)
        {
            break;
        }
    }
    get_g2();
    radioCorrection();
    get_dg();
    construct_matrices();
    precision();
}

void matching::get_result()
{
    double a0 = X.getMatrix_ele(2,0);
    double a1 = X.getMatrix_ele(3,0);
    double a2 = X.getMatrix_ele(4,0);
    double b0 = X.getMatrix_ele(5,0);
    double b1 = X.getMatrix_ele(6,0);
    double b2 = X.getMatrix_ele(7,0);
    double h0 = X.getMatrix_ele(0,0);
    double h1 = X.getMatrix_ele(1,0); //获取形变改正参数
    distortion_window = cv::Mat(left_window.size(), CV_32F, cv::Scalar(0.0f)); //初始化新窗口
    int k = (int)(window_size/2); //窗口大小的一半
    for(int i = -k; i<k+1; i++)
    {
        for(int j = -k; j<k+1;j++)
        {
            int x = rightx + i;
            int y = righty + j; //变形前的像素点坐标
            double x2 = a0 + a1*x + a2*y;
            double y2 = b0 + b1*x + b2*y; //变形后的像素点坐标
            if(x2 < rightx - k || x2 > rightx + k || y2 < righty - k || y2 > righty + k)
            {
                continue;
            }
            distortion_window.at<float>( std::round(y2 - righty + k), std::round(x2 - rightx + k)) = sample_img(right_img, x2, y2); //获取原窗口位置下变形后的窗口内容
        }
    }

    //显示窗口内容
    cv::Mat right_win_show;
    cv::normalize(distortion_window, right_win_show, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Distortion",right_win_show);
    cv::waitKey(0);

    //distortion_window.convertTo(distortion_window, CV_32F, 1.0 / 255.0);
    //cv::Mat img_8u;
    //distortion_window.convertTo(distortion_window, CV_8U, 255.0);
    //cv::imwrite(std::string("dist") + ".jpg", distortion_window);
}

cv::Mat matching::get_left_window()
{
    return left_window.clone();
}

cv::Mat matching::get_right_window()
{
    return right_window.clone();
}

void matching::get_matched_points(std::string savepath)
{
    CorrelationMatch match;
    match.Calculate(left_img, right_img, matching_window_size, ncc_threshold);
    match.saveResult(savepath);
}

void matching::set_matching_params(int window_size, double threshold)
{
    matching_window_size = window_size;
    ncc_threshold = threshold;
}

double matching::get_matched_x()
{
    return new_rightx;
}

double matching::get_matched_y()
{
    return new_righty;
}

double matching::get_delta0()
{
    return delta0;
}

double matching::get_SNR()
{
    return SNR;
}
    
double matching::get_rho()
{
    return rho;
}

double matching::get_deltag()
{
    return deltag;
}
    
double matching::get_deltag_()
{
    return deltag_;
}
    
double matching::get_deltax()
{
    return deltax;
}

double matching::get_h0()
{
    return X.getMatrix_ele(0,0);
}

double matching::get_h1()
{
    return X.getMatrix_ele(1,0);
}
    
double matching::get_a0()
{
    return X.getMatrix_ele(2,0);
}
    
double matching::get_a1()
{
    return X.getMatrix_ele(3,0);
}
    
double matching::get_a2()
{
    return X.getMatrix_ele(4,0);
}
    
double matching::get_b0()
{
    return X.getMatrix_ele(5,0);
}
    
double matching::get_b1()
{
    return X.getMatrix_ele(6,0);
}
    
double matching::get_b2()
{
    return X.getMatrix_ele(7,0);
}

#ifdef HAS_CUDA
#include "correlation_cuda.h"
#include "extract_cuda.h"
#include "matching_cuda.h"
#include "cuda_common.h"

int matching::gpu_device_count()
{
    return get_cuda_device_count();
}

std::vector<std::tuple<int, int, double, double>>
matching::batch_adjust_gpu(
    const std::vector<std::pair<int, int>>& left_centers,
    const std::vector<std::pair<int, int>>& right_centers,
    int window_size, double d_corr_threshold, int max_iterations
)
{
    auto results = matching_adjust_gpu(
        left_img.data, left_img.cols, left_img.rows,
        right_img.data, right_img.cols, right_img.rows,
        left_centers, right_centers,
        window_size, (float)d_corr_threshold, max_iterations
    );
    std::vector<std::tuple<int, int, double, double>> corrected;
    for (size_t i = 0; i < results.size(); i++)
    {
        double mx = results[i].matched_x;
        double my = results[i].matched_y;
        if (std::isnan(mx) || std::isnan(my)) continue;
        if (mx < 0.0 || my < 0.0) continue;
        if (mx >= (double)right_img.cols || my >= (double)right_img.rows) continue;
        if (results[i].iterations >= max_iterations && !results[i].converged) continue;

        corrected.push_back({
            left_centers[i].first, left_centers[i].second,
            results[i].matched_x, results[i].matched_y
        });
    }
    return corrected;
}
#endif