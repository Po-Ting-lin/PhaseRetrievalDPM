#include "phase_retriever.cuh"

__global__ void realToComplex(uchar* src, fComplex* dst, int size) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= size) return;
    dst[x].x = src[x];
    dst[x].y = 0.0f;
}

__global__ void complexToMagnitude(fComplex* src, float* dst, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= width || y >= height) return;
    dst[y * width + x] = 20.0f * logf(sqrtf(powf(src[y * width + x].x, 2.0f) + powf(src[y * width + x].y, 2.0f)));
}

__global__ void copyInterferenceComponentRoughly(float* src, float* dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;
    int hw = dstWidth / 2;
    int hh = dstHeight / 2;
    int srcX = x - hw;
    int srcY = y - hh + 5 * srcHeight / 8;
    if (srcX < 0) {
        srcX = srcWidth + srcX;
    }
    dst[y * dstWidth + x] = src[srcY * srcWidth + srcX];
}

__global__ void copyInterferenceComponentDebug(float* src, float* dst, int centerX, int centerY, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;
    int hw = dstWidth / 2;
    int hh = dstHeight / 2;
    int srcX = x - hw + centerX;
    int srcY = y - hh + centerY;
    if (srcX < 0) {
        srcX = srcWidth + srcX;
    }
    int dstX = -1;
    int dstY = -1;
    if (x < hw && y < hh) {
         dstX = x + hw;
         dstY = y + hh;
    }
    else if (x >= hw && y < hh) {
         dstX = x - hw;
         dstY = y + hh;
    }
    else if (x < hw && y >= hh) {
         dstX = x + hw;
         dstY = y - hh;
    }
    else {
         dstX = x - hw;
         dstY = y - hh;
    }
    dst[dstY * dstWidth + dstX] = src[srcY * srcWidth + srcX];
}

__global__ void copyInterferenceComponent(fComplex* src, fComplex* dst, int centerX, int centerY, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;
    int hw = dstWidth / 2;
    int hh = dstHeight / 2;
    int srcX = x - hw + centerX;
    int srcY = y - hh + centerY;
    if (srcX < 0) {
        srcX = srcWidth + srcX;
    }
    int dstX = -1;
    int dstY = -1;
    if (x < hw && y < hh) {
        dstX = x + hw;
        dstY = y + hh;
    }
    else if (x >= hw && y < hh) {
        dstX = x - hw;
        dstY = y + hh;
    }
    else if (x < hw && y >= hh) {
        dstX = x + hw;
        dstY = y - hh;
    }
    else {
        dstX = x - hw;
        dstY = y - hh;
    }
    dst[dstY * dstWidth + dstX].x = src[srcY * srcWidth + srcX].x;
    dst[dstY * dstWidth + dstX].y = src[srcY * srcWidth + srcX].y;
}

__global__ void applyArcTan(fComplex* src, float* dst, int srcWidth, int srcHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= srcWidth || y >= srcHeight) return;
    dst[y * srcWidth + x] = atan2f(src[y * srcWidth + x].y, src[y * srcWidth + x].x);
}