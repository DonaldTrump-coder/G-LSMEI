# G-LSMEI: GPU-Accelerated Least-Square Matching Refiner for Epipolar Image Pairs
[![GitHub Repo stars](https://img.shields.io/github/stars/DonaldTrump-coder/G-LSMEI?style=social)](https://github.com/DonaldTrump-coder/G-LSMEI/)
[![GitHub Repo forks](https://img.shields.io/github/forks/DonaldTrump-coder/G-LSMEI?style=social)](https://github.com/DonaldTrump-coder/G-LSMEI/)
[![PyPI version](https://img.shields.io/pypi/v/glsmei?color=blue)](https://pypi.org/project/glsmei/)
[![Project Page](https://img.shields.io/badge/Project_Page-G--LSMEI-brightgreen)](https://donaldtrump-coder.github.io/G-LSMEI/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat)](http://www.apache.org/licenses/)
![Last Commit](https://img.shields.io/github/last-commit/DonaldTrump-coder/G-LSMEI?color=green)
![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%20Windows%2011-lightgrey)<br>
A project of the sub-pixel Least-Square Matching Refining Algorithm for **Epipolar-Rectified Stereo Image Pairs** in window-size areas. The classic algorithm is proposed by Ackermann in [*DIGITAL IMAGE CORRELATION: PERFORMANCE AND POTENTIAL APPLICATION IN PHOTOGRAMMETRY*](https://doi.org/10.1111/j.1477-9730.1984.tb00505.x) (1984), where the correspondence search reduces to a one-dimensional problem along epipolar lines, enabling both fast convergence and high accuracy.<br><br>
We provide the source code in C++ and a Python API library built upon it. A PyQt Demo Application and the CUDA-based GPU-accelerated version is also provided.<br>
Up to now we have successfully tested the C++ and Python on *Windows 10* and *Windows 11*.<br><br>
**[Project Page](https://donaldtrump-coder.github.io/G-LSMEI/)**<br>
**Contributors**: [Haojun Tang](https://donaldtrump-coder.github.io/), [Jiahao Zhou](https://github.com/Jeiluo)<br>
**Acknowledgements**: Thanks to the guidance of [Yunsheng Zhang](https://faculty.csu.edu.cn/zhangyunsheng1/zh_CN/index.htm) from [Central South University](https://www.csu.edu.cn/).

## About the Project
### Environment
<img src="https://raw.githubusercontent.com/DonaldTrump-coder/G-LSMEI/main/docs/images/Environments.png" width="40%" />

### Application Structure
<img src="https://raw.githubusercontent.com/DonaldTrump-coder/G-LSMEI/main/docs/images/Structures.png" width="55%" />

### CUDA Executions
<img src="https://raw.githubusercontent.com/DonaldTrump-coder/G-LSMEI/main/docs/images/cudastream.png" width="75%" />

## Some Results
### Matching Results
|Left Image|Right Image|
|:--:|:--:|
|<img src="https://raw.githubusercontent.com/DonaldTrump-coder/G-LSMEI/main/docs/images/left_marked.jpg" width=100%>|<img src="https://raw.githubusercontent.com/DonaldTrump-coder/G-LSMEI/main/docs/images/right_marked.jpg" width=100%>|

### Time Costs
|Setups| CPU | GPU |
|:-------:|:-------:|:-------:|
|Time (sec)| 332.27 | 0.621 |

Improved by 500+ times in our GPU implementation!

## Core Algorithm tested in C++ (only CPU-based)
**Prerequisites**<br>
**MSVC** (Visual Studio 2022, with C++ CMake tools), **CMake** ≥ 3.20, and the source code:
```
git clone https://github.com/DonaldTrump-coder/G-LSMEI --recursive
```

**Build for the Code**<br>
In the project directory, run the following commands:<br>
```
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
.\Release\leastsquares_matching.exe
```
The output is from the main function in `core\src\test.cpp`<br>
Use the C++ source code for testing the algorithm.

## Image Matching Application Deployment (CPU and GPU support)
**Prerequisites**<br>
You also need **MSVC** and **CMake**, **conda** (or any Python environment), as well as **nvcc** for CUDA (Already successfully tested on CUDA 12.4). Install them and start to build!<br><br>
**1. Install the Requirement Packages for Python**<br>
In the project directory, run<br>
```
conda create -n matching python=3.11
conda activate matching
pip install -r requirements.txt
```

**2. Build the C++ and CUDA Source for Python Application**<br>
```
cd python
python setup.py bdist_wheel
python -m pip install (Get-ChildItem dist\glsmei-*.whl).FullName --force-reinstall
cd ..
```
Then the installation of `glsmei` is done.<br><br>
**3. Use the Application in Python**<br>
run
```
python main.py
```

## API Reference
**Installation from pip** (If your Python and CUDA environment is supported): `pip install glsmei`<br>
**Verify installation**: `python -c "import lsmatching; print(lsmatching.__version__)"`
### Matching in CPU
```python
Matching(left_image_path: str, right_image_path: str)
```
| APIs | Params | Returns | Description |
|------|------|------|------|
| `set_params(windowsize, d_corr_threshold)` | `int=15`, `float=0.04` | — | Set template window size and correlation change threshold |
| `set_matching_params(windowsize, corr_threshold)` | `int=3`, `float=0.7` | — | Set initial matching parameters |
| `set_centers(x1, y1, x2, y2)` | `int` × 4 | — | Set initial conjugate points on left/right images |
| `calculate()` | — | — | Run single-point least-squares matching |
| `get_left_window()` | — | `np.ndarray` | Get the left image matching window |
| `get_right_window()` | — | `np.ndarray` | Get the right image matching window |
| `get_matched_points(savepath)` | `str` | — | Export matching results |
| `get_matched_x()` | — | `float` | Matched point x-coordinate |
| `get_matched_y()` | — | `float` | Matched point y-coordinate |
| `get_delta0()` | — | `float` | Standard error of unit weight |
| `get_deltag()` | — | `float` | Standard error of window |
| `get_deltax()` | — | `float` | Standard error of parameters |
| `get_SNR()` | — | `float` | SNR |
| `get_h0()` / `get_h1()` | — | `float` | Radiometric distortion parameters |
| `get_a0()` / `get_a1()` / `get_a2()` | — | `float` | Affine parameters (left → right, x-direction) |
| `get_b0()` / `get_b1()` / `get_b2()` | — | `float` | Affine parameters (left → right, y-direction) |
| `gpu_device_count()` | — | `int` | Number of available GPUs |

### Matching in GPU
```python
batch_adjust_gpu(window_size=15, d_corr=0.04, max_iter=20, matching_wsize=3, corr_threshold=0.7, savepath=None)
```
GPU batch pipeline: Feature Extraction → Correlation Matching → Least-Squares Refinement<br>
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | `int` | 15 | Least-squares refinement template window size |
| `d_corr` | `float` | 0.04 | Correlation change threshold (convergence criterion) |
| `max_iter` | `int` | 20 | Maximum iteration count |
| `matching_wsize` | `int` | 3 | Initial correlation matching template window size |
| `corr_threshold` | `float` | 0.7 | Initial matching correlation threshold |
| `savepath` | `str` | `None` | Output save path |

## License
This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) details.

## Citation
If you use our work or our data in your research, please cite:
```bibtex
@misc{Tang2025GLSMEI,
  title        = {G-LSMEI: GPU-Accelerated Least-Square Matching Refiner for Epipolar Image Pairs},
  author       = {Haojun Tang and Jiahao Zhou},
  year         = {2025},
  howpublished = {\url{https://github.com/DonaldTrump-coder/LeastSquares_Matching}},
  note         = {Version 1.0.2. Apache License 2.0}
}
```
