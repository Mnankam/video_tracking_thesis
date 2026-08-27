#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>


int main()
{
    std::cout << "========================================\n";
    std::cout << "OpenCV / CUDA capability check\n";
    std::cout << "========================================\n";

    std::cout << "OpenCV version: "
              << CV_VERSION
              << "\n\n";

    const int cudaDevices =
        cv::cuda::getCudaEnabledDeviceCount();

    std::cout << "CUDA-enabled devices visible to OpenCV: "
              << cudaDevices
              << "\n";


    if (cudaDevices <= 0)
    {
        std::cout
            << "\nNo CUDA-enabled OpenCV device is available.\n"
            << "Possible reasons:\n"
            << "  - OpenCV was built without CUDA\n"
            << "  - no GPU was allocated\n"
            << "  - CUDA drivers are not visible\n";

        return 0;
    }


    for (int i = 0; i < cudaDevices; ++i)
    {
        cv::cuda::DeviceInfo device(i);

        std::cout << "\nDevice " << i << "\n";
        std::cout << "Name: "
                  << device.name()
                  << "\n";

        std::cout << "Compute capability: "
                  << device.majorVersion()
                  << "."
                  << device.minorVersion()
                  << "\n";

        std::cout << "Total memory: "
                  << device.totalMemory() / (1024.0 * 1024.0)
                  << " MB\n";

        std::cout << "Compatible: "
                  << (device.isCompatible() ? "yes" : "no")
                  << "\n";
    }


    std::cout << "\n========================================\n";

    return 0;
}