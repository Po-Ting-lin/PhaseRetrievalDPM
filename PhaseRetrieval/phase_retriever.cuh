#pragma once
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <cufft.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error.cuh"
#include "parameters.h"
#include "utils.h"


extern "C" { 
    ExportDll void PhaseRetriever(uchar* sp, uchar* bg, float* dst, int width, int height, int spx, int spy, int bgx, int bgy);
}
void imageRetriever(uchar* src, float*& dst, PhaseRetrieverInfo& info, bool isSp);
void getWrappedImage(PhaseRetrieverInfo& info, bool isSp);
void getUnwrappedImage(PhaseRetrieverInfo& info, bool isSp);
__global__ void realToComplex(uchar* src, fComplex* dst, int size);
__global__ void complexToMagnitude(fComplex* src, float* dst, int width, int height);
__global__ void copyInterferenceComponentRoughly(float* src, float* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
__global__ void copyInterferenceComponentDebug(float* src, float* dst, int centerX, int centerY, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
__global__ void copyInterferenceComponent(fComplex* src, fComplex* dst, int centerX, int centerY, int srcWidth, int srcHeight, int dstWidth, int dstHeight);
__global__ void applyArcTan(fComplex* src, float* dst, int srcWidth, int srcHeight);
__global__ void applyDifference(float* src, float* dxp, float* dyp, int srcWidth, int srcHeight);
__global__ void applySum(float* dxp, float* dyp, float* sumC, float* divider, float tx, float ty, int srcWidth, int srcHeight);
__global__ void max_idx_kernel(const float* data, const int dsize, int* result, float* blk_vals, int* blk_idxs, int* blk_num);
__device__ void frequencyShift(int x, int y, int hw, int hh, int& dstX, int& dstY);