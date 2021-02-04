#pragma once
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include "parameters.h"
#include "utils.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

typedef unsigned int uint;

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
    float x;
    float y;
} fComplex;
#endif

struct PhaseRetrieverInfo
{
    cv::Mat* Image;
    float* WrappedImage;
    int Width;
    int Height;
    int CroppedWidth;
    int CroppedHeight;
    int NumberOfRealElements;
    int NumberOfCropElements;
    dim3* Grids;
    dim3* CroppedGrids;
    dim3* Blocks;
};

void processPhaseRetriever(cv::Mat& src);
void getWrappedImage(PhaseRetrieverInfo& info);
void getUnwrappedImage(PhaseRetrieverInfo& info);
__global__ void realToComplex(uchar* src, fComplex* dst, int size);
__global__ void complexToMagnitude(fComplex* src, float* dst, int width, int height);
__global__ void copyInterferenceComponentRoughly(float* src, float* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
__global__ void copyInterferenceComponentDebug(float* src, float* dst, int centerX, int centerY, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
__global__ void copyInterferenceComponent(fComplex* src, fComplex* dst, int centerX, int centerY, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
__global__ void applyArcTan(fComplex* src, float* dst, int srcWidth, int srcHeight);
