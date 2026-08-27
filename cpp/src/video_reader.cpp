#include "video_reader.hpp"

#include <stdexcept>


VideoReader::VideoReader(
    const std::string& videoPath
)
{
    capture_.open(videoPath);

    if (!capture_.isOpened())
    {
        throw std::runtime_error(
            "Could not open video: " + videoPath
        );
    }
}


bool VideoReader::read(
    cv::Mat& frame
)
{
    return capture_.read(frame);
}


bool VideoReader::isOpened() const
{
    return capture_.isOpened();
}


double VideoReader::fps() const
{
    return capture_.get(
        cv::CAP_PROP_FPS
    );
}


long long VideoReader::frameCount() const
{
    return static_cast<long long>(
        capture_.get(
            cv::CAP_PROP_FRAME_COUNT
        )
    );
}


int VideoReader::width() const
{
    return static_cast<int>(
        capture_.get(
            cv::CAP_PROP_FRAME_WIDTH
        )
    );
}


int VideoReader::height() const
{
    return static_cast<int>(
        capture_.get(
            cv::CAP_PROP_FRAME_HEIGHT
        )
    );
}