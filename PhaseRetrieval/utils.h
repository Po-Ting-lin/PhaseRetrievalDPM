#pragma once
#define TILE_DIM 32

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b)
{
    return (a % b != 0) ? (a - a % b + b) : a;
}

// Rounding up the FFT dimensions to the next power of 2,
// unless the dimension would exceed 1024, 
// in which case it's rounded up to the next multiple of 512.
// Reference: zchee -- https://github.com/zchee/cuda-sample
static int snapTransformSize(int dataSize) {
    int hiBit;
    unsigned int lowPOT, hiPOT;

    dataSize = iAlignUp(dataSize, 16);
    for (hiBit = 31; hiBit >= 0; hiBit--) {
        if (dataSize & (1U << hiBit)) {
            break;
        }
    }
    lowPOT = 1U << hiBit;

    if (lowPOT == (unsigned int)dataSize)
        return dataSize;
    hiPOT = 1U << (hiBit + 1);
    if (hiPOT <= 1024)
        return hiPOT;
    else
        return iAlignUp(dataSize, 512);
}

static void printTime(std::chrono::system_clock::time_point t1, std::chrono::system_clock::time_point t2, std::string name) {
    std::chrono::duration<double> time_lapse = t2 - t1;
    std::cout << name << " time consume: " << time_lapse.count() << " s" << std::endl;
}

static void displayImage(float* src, int width, int height, bool mag) {
    cv::Mat Out(height, width, CV_32F, src, width * sizeof(float));
    cv::Mat Out2(height, width, CV_8U);
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    minMaxLoc(Out, &minVal, &maxVal, &minLoc, &maxLoc);

    //std::cout << "max: " << maxVal << "; min:" << minVal << std::endl;

    for (int i = 0; i < width * height; i++) {
        Out2.data[i] = 255 * ((src[i] - (float)minVal) / ((float)maxVal - (float)minVal));
    }

    cv::Mat Outmag;
    if (mag) {
        cv::resize(Out2, Outmag, cv::Size(Out2.cols / 4, Out2.rows / 4), 5, 5);
    }
    else {
        Out2.copyTo(Outmag);
    }
    namedWindow("here", cv::WINDOW_AUTOSIZE);
    cv::imshow("here", Outmag);
    cv::waitKey(0);
}