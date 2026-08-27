#pragma once

#include <opencv2/opencv.hpp>


class MaskPostprocessor
{
public:

    MaskPostprocessor(
        int minimumArea = 100,
        int openingKernel = 3,
        int closingKernel = 5
    );


    cv::Mat process(
        const cv::Mat& inputMask
    ) const;


private:

    int minimumArea_;

    int openingKernel_;

    int closingKernel_;
};