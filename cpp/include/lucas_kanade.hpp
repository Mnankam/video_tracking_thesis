#pragma once

#include <opencv2/opencv.hpp>

#include <vector>


struct LKResult
{
    int pointId;

    cv::Point2f previousPoint;

    cv::Point2f currentPoint;

    float dx;

    float dy;

    double displacement;

    double fbError;

    bool valid;
};


class LucasKanadeTracker
{
public:

    LucasKanadeTracker(
        cv::Size windowSize = cv::Size(21, 21),
        int maxLevel = 3,
        double maxJumpPx = 5.0,
        double maxFbError = 2.0
    );


    std::vector<LKResult> track(
        const cv::Mat& previousGray,
        const cv::Mat& currentGray,
        const std::vector<cv::Point2f>& points
    ) const;


private:

    cv::Size windowSize_;

    int maxLevel_;

    double maxJumpPx_;

    double maxFbError_;
};