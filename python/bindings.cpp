#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "matching.h"
#include "Correlation.h"

namespace py = pybind11;

py::array mat_to_numpy(const cv::Mat &mat)
{
    if (mat.empty())
        throw std::runtime_error("mat_to_numpy: empty cv::Mat");

    int type = mat.type();
    int depth = CV_MAT_DEPTH(type);

    if (depth == CV_8U)
    {
        return py::array(
            py::buffer_info(
                mat.data,
                sizeof(uint8_t),
                py::format_descriptor<uint8_t>::format(),
                2,
                { mat.rows, mat.cols },
                { (size_t)mat.step, sizeof(uint8_t) }
            )
        );
    }
    else if (depth == CV_32F)
    {
        return py::array(
            py::buffer_info(
                mat.data,
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                { mat.rows, mat.cols },
                { (size_t)mat.step, sizeof(float) }
            )
        );
    }
    else
    {
        throw std::runtime_error("mat_to_numpy: unsupported cv::Mat type (only CV_8U & CV_32F)");
    }
}

PYBIND11_MODULE(_lsmatching, m)
{
    m.doc() = "Least squares matching module";
    py::class_<matching>(m, "Matching")
        .def(py::init<std::string, std::string>())
        .def("set_params", &matching::set_params)
        .def("set_matching_params", &matching::set_matching_params)
        .def("set_centers", &matching::set_centers)
        .def("calculate", &matching::calculate)
        .def("get_left_window", [](matching &self) {
            return mat_to_numpy(self.get_left_window());
        })
        .def("get_right_window", [](matching &self) {
            return mat_to_numpy(self.get_right_window());
            })
        .def("get_matched_points", &matching::get_matched_points)
        .def("get_matched_x", &matching::get_matched_x)
        .def("get_matched_y", &matching::get_matched_y)
        .def("get_delta0", &matching::get_delta0)
        .def("get_SNR", &matching::get_SNR)
        .def("get_rho", &matching::get_rho)
        .def("get_deltag", &matching::get_deltag)
        .def("get_deltag_", &matching::get_deltag_)
        .def("get_deltax", &matching::get_deltax)
        .def("get_h0", &matching::get_h0)
        .def("get_h1", &matching::get_h1)
        .def("get_a0", &matching::get_a0)
        .def("get_a1", &matching::get_a1)
        .def("get_a2", &matching::get_a2)
        .def("get_b0", &matching::get_b0)
        .def("get_b1", &matching::get_b1)
        .def("get_b2", &matching::get_b2)
        .def("gpu_device_count", &matching::gpu_device_count)
        .def("batch_adjust_gpu", &matching::batch_adjust_gpu)
        ;
    py::class_<CorrelationMatch>(m, "CorrelationMatch")
        .def(py::init<>())
        .def("calculate_gpu", [](CorrelationMatch &cm,
                                  const std::string& left_path,
                                  const std::string& right_path,
                                  int window_size, double ncc_threshold)
                                  {
            cv::Mat left  = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
            cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
            cm.CalculateGPU(left, right, window_size, ncc_threshold);
        })
        .def("save_result", &CorrelationMatch::saveResult)
        .def("get_left_same", [](CorrelationMatch &cm) {
            py::list out;
            for (const auto& p : cm.getLeftSame())
                out.append(py::make_tuple(p.x, p.y));
            return out;
        })
        .def("get_rig_same", [](CorrelationMatch &cm) {
            py::list out;
            for (const auto& p : cm.getRigSame())
                out.append(py::make_tuple(p.x, p.y));
            return out;
        })
        ;
}