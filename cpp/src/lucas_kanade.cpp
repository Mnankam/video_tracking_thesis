#include "lucas_kanade.hpp"

#include <cmath>


LucasKanadeTracker::LucasKanadeTracker(
    cv::Size windowSize,
    int maxLevel,
    double maxJumpPx,
    double maxFbError
)
    :
    windowSize_(windowSize),
    maxLevel_(maxLevel),
    maxJumpPx_(maxJumpPx),
    maxFbError_(maxFbError)
{
}


std::vector<LKResult>
LucasKanadeTracker::track(
    const cv::Mat& previousGray,
    const cv::Mat& currentGray,
    const std::vector<cv::Point2f>& points
) const
{
    std::vector<cv::Point2f> forwardPoints;

    std::vector<unsigned char> forwardStatus;

    std::vector<float> forwardError;


    cv::calcOpticalFlowPyrLK(
        previousGray,
        currentGray,
        points,
        forwardPoints,
        forwardStatus,
        forwardError,
        windowSize_,
        maxLevel_
    );


    std::vector<cv::Point2f> backwardPoints;

    std::vector<unsigned char> backwardStatus;

    std::vector<float> backwardError;


    cv::calcOpticalFlowPyrLK(
        currentGray,
        previousGray,
        forwardPoints,
        backwardPoints,
        backwardStatus,
        backwardError,
        windowSize_,
        maxLevel_
    );


    std::vector<LKResult> results;

    results.reserve(
        points.size()
    );


    for (
        std::size_t i = 0;
        i < points.size();
        ++i
    )
    {
        const float dx =
            forwardPoints[i].x
            -
            points[i].x;

        const float dy =
            forwardPoints[i].y
            -
            points[i].y;


        const double displacement =
            std::sqrt(
                dx * dx
                +
                dy * dy
            );


        const double fbError =
            cv::norm(
                backwardPoints[i]
                -
                points[i]
            );


        const bool valid =
            forwardStatus[i]
            &&
            backwardStatus[i]
            &&
            std::isfinite(displacement)
            &&
            std::isfinite(fbError)
            &&
            displacement <= maxJumpPx_
            &&
            fbError <= maxFbError_;


        LKResult result;

        result.pointId =
            static_cast<int>(i);

        result.previousPoint =
            points[i];

        result.currentPoint =
            forwardPoints[i];

        result.dx =
            dx;

        result.dy =
            dy;

        result.displacement =
            displacement;

        result.fbError =
            fbError;

        result.valid =
            valid;


        results.push_back(
            result
        );
    }


    return results;
}