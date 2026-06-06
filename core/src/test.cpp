#include "matching.h"
#include <iostream>

int main()
{
    std::string left_path = "../data/uavl_epi.jpg";
    std::string right_path = "../data/uavr_epi.jpg";
    matching match(left_path, right_path);
    match.set_params(45, 0.008);
    match.set_centers(2296,2127,1773,2127);
    match.disp_windows();
    match.calculate();
    match.get_result();
    return 0;
}