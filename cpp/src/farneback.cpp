#include "farneback.hpp"

#include <cmath>


FarnebackTracker::FarnebackTracker(
    double pyrScale,
    int levels,
    int windowSize,
    int iterations,
    int polyN,
    double polySigma
)
    :
    pyrScale_(pyrScale),
    levels_(levels),
    windowSize_(windowSize),
    iterations_(iterations),
    polyN_(polyN),
    polySigma_(polySigma)
{
}


FarnebackResult
FarnebackTracker::compute(
    const cv::Mat& previousGray,
    const cv::Mat& currentGray,
    const cv::Rect& roi
) const
{
    cv::Mat flow;


    cv::calcOpticalFlowFarneback(
        previousGray,
        currentGray,
        flow,
        pyrScale_,
        levels_,
        windowSize_,
        iterations_,
        polyN_,
        polySigma_,
        0
    );


    const cv::Rect imageRect(
        0,
        0,
        flow.cols,
        flow.rows
    );


    const cv::Rect validRoi =
        roi & imageRect;


    if (
        validRoi.width <= 0
        ||
        validRoi.height <= 0
    )
    {
        return {
            0.0,
            0.0,
            0.0,
            false
        };
    }


    cv::Mat roiFlow =
        flow(validRoi);


    std::vector<cv::Mat> channels;

    cv::split(
        roiFlow,
        channels
    );


    const double dx =
        cv::mean(
            channels[0]
        )[0];

    const double dy =
        cv::mean(
            channels[1]
        )[0];


    const double magnitude =
        std::sqrt(
            dx * dx
            +
            dy * dy
        );


    return {
        dx,
        dy,
        magnitude,
        std::isfinite(dx)
        &&
        std::isfinite(dy)
        &&
        std::isfinite(magnitude)
    };
}