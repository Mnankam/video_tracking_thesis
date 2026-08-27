#pragma once

#include <opencv2/opencv.hpp>


struct FarnebackResult
{
    double meanDx;

    double meanDy;

    double magnitude;

    bool valid;
};


class FarnebackTracker
{
public:

    FarnebackTracker(
        double pyrScale = 0.5,
        int levels = 3,
        int windowSize = 15,
        int iterations = 3,
        int polyN = 5,
        double polySigma = 1.2
    );


    FarnebackResult compute(
        const cv::Mat& previousGray,
        const cv::Mat& currentGray,
        const cv::Rect& roi
    ) const;


private:

    double pyrScale_;

    int levels_;

    int windowSize_;

    int iterations_;

    int polyN_;

    double polySigma_;
};