#pragma once

#include <opencv2/opencv.hpp>

#include <string>


class VideoReader
{
public:

    explicit VideoReader(
        const std::string& videoPath
    );

    bool read(
        cv::Mat& frame
    );

    bool isOpened() const;

    double fps() const;

    long long frameCount() const;

    int width() const;

    int height() const;


private:

    cv::VideoCapture capture_;
};