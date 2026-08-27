#include "postprocessing.hpp"


MaskPostprocessor::MaskPostprocessor(
    int minimumArea,
    int openingKernel,
    int closingKernel
)
    :
    minimumArea_(minimumArea),
    openingKernel_(openingKernel),
    closingKernel_(closingKernel)
{
}


cv::Mat
MaskPostprocessor::process(
    const cv::Mat& inputMask
) const
{
    cv::Mat mask;


    if (
        inputMask.type()
        !=
        CV_8UC1
    )
    {
        inputMask.convertTo(
            mask,
            CV_8UC1
        );
    }
    else
    {
        mask =
            inputMask.clone();
    }


    cv::threshold(
        mask,
        mask,
        0,
        255,
        cv::THRESH_BINARY
    );


    const cv::Mat openingKernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(
                openingKernel_,
                openingKernel_
            )
        );


    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_OPEN,
        openingKernel
    );


    const cv::Mat closingKernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(
                closingKernel_,
                closingKernel_
            )
        );


    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_CLOSE,
        closingKernel
    );


    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;


    const int count =
        cv::connectedComponentsWithStats(
            mask,
            labels,
            stats,
            centroids,
            8,
            CV_32S
        );


    cv::Mat result =
        cv::Mat::zeros(
            mask.size(),
            CV_8UC1
        );


    for (
        int label = 1;
        label < count;
        ++label
    )
    {
        const int area =
            stats.at<int>(
                label,
                cv::CC_STAT_AREA
            );


        if (
            area >= minimumArea_
        )
        {
            result.setTo(
                255,
                labels == label
            );
        }
    }


    return result;
}
