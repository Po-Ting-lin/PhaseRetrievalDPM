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
    int i = y * width + x;
    const float floatTab[2] = { src[i].x , src[i].y };
    dst[i] = 20.0f * logf(normf(2, floatTab));
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
    frequencyShift(x, y, hw, hh, dstX, dstY);
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
    frequencyShift(x, y, hw, hh, dstX, dstY);
    dst[dstY * dstWidth + dstX].x = src[srcY * srcWidth + srcX].x;
    dst[dstY * dstWidth + dstX].y = src[srcY * srcWidth + srcX].y;
}

__global__ void applyArcTan(fComplex* src, float* dst, int srcWidth, int srcHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= srcWidth || y >= srcHeight) return;
    dst[y * srcWidth + x] = atan2f(src[y * srcWidth + x].y, src[y * srcWidth + x].x);
}

__global__ void applyDifference(float* src, float* dxp, float* dyp, int srcWidth, int srcHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int i = y * srcWidth + x;
    if (x >= srcWidth || y >= srcHeight) return;
    float dx = 0.0f;
    float dy = 0.0f;
    if (x + 1 < srcWidth) {
        dx = src[y * srcWidth + x + 1] - src[i];
    }
    if (y + 1 < srcHeight) {
        dy = src[(y + 1) * srcWidth + x] - src[i];
    }
    float sign_value_x = signbit(dx) ? -1.0f : 1.0f; // branching, which can be optimized!
    float sign_value_y = signbit(dy) ? -1.0f : 1.0f; // branching, which can be optimized!
    dxp[i] = dx - PI2 * sign_value_x * floorf((fabsf(dx) + PI) / PI2);
    dyp[i] = dy - PI2 * sign_value_y * floorf((fabsf(dy) + PI) / PI2);
}

__global__ void applySum(float* dxp, float* dyp, float* sumC, float* divider, float tx, float ty, int srcWidth, int srcHeight) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int i = y * srcWidth + x;
    if (x >= srcWidth || y >= srcHeight) return;
    float c1 = 0.0f;
    float c2 = 0.0f;
    float c3 = 0.0f;
    float c4 = 0.0f;
    if (x == 0) {
        c1 = dxp[i];
    }
    else if (x == srcWidth - 1) {
        c2 = dxp[y * srcWidth + x - 1];
    }
    else {
        c1 = dxp[i];
        c2 = dxp[y * srcWidth + x - 1];
    }
    if (y == 0) {
        c3 = dyp[i];
    }
    else if (y == srcHeight - 1) {
        c4 = dyp[(y - 1) * srcWidth + x];
    }
    else {
        c3 = dyp[i];
        c4 = dyp[(y - 1) * srcWidth + x];
    }
    sumC[i] = c1 - c2 + c3 - c4;
    divider[i] = 2.0f * cosf(tx * x) + 2.0f * cosf(ty * y) - 4.0f;
}

__device__ void frequencyShift(int x, int y, int hw, int hh, int& dstX, int& dstY) {
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
}
