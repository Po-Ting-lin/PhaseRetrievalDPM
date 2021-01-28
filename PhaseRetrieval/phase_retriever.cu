#include "phase_retriever.cuh"

__global__ void padKernelKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
) {
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (y < kernelH && x < kernelW)
    {
        int ky = y - kernelY;
        if (ky < 0)
        {
            ky += fftH;
        }
        int kx = x - kernelX;
        if (kx < 0)
        {
            kx += fftW;
        }
        d_Dst[ky * fftW + kx] = d_Src[y * kernelW + x];
    }
}

__global__ void padDataClampToBorderKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
) {
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int borderH = dataH + kernelY;
    const int borderW = dataW + kernelX;

    if (y < fftH && x < fftW) {
        int dy, dx;
        if (y < dataH) {
            dy = y;
        }
        if (x < dataW) {
            dx = x;
        }
        if (y >= dataH && y < borderH) {
            dy = dataH - 1;
        }
        if (x >= dataW && x < borderW) {
            dx = dataW - 1;
        }
        if (y >= borderH) {
            dy = 0;
        }
        if (x >= borderW) {
            dx = 0;
        }
        d_Dst[y * fftW + x] = d_Src[dy * dataW + dx];
    }
}

__global__ void modulateAndNormalizeKernel(
    fComplex* d_Dst,
    fComplex* d_DataSrc,
    fComplex* d_KernelSrc,
    int dataSize,
    float c
) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= dataSize) return;

    fComplex a = d_KernelSrc[i];
    fComplex b = d_DataSrc[i];
    fComplex d;
    mulAndScaleModified(a, b, c, d);
    d_Dst[i] = d;
}

void padKernel(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
) {
    assert(d_Src != d_Dst);
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(kernelW, block.x), iDivUp(kernelH, block.y));
    padKernelKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
        );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
    cudaDeviceSynchronize();
}

void padDataClampToBorder(
    float* d_Dst,
    float* d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
) {
    assert(d_Src != d_Dst);
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(iDivUp(fftW, block.x), iDivUp(fftH, block.y));
    padDataClampToBorderKernel << <grid, block >> > (
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
        );
    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

void convolutionClampToBorderCPU(
    float* h_Result,
    float* h_Data,
    float* h_Kernel,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
) {
    for (int y = 0; y < dataH; y++) {
        for (int x = 0; x < dataW; x++) {
            double sum = 0;
            for (int ky = -(kernelH - kernelY - 1); ky <= kernelY; ky++) {
                for (int kx = -(kernelW - kernelX - 1); kx <= kernelX; kx++)
                {
                    int dy = y + ky;
                    int dx = x + kx;

                    if (dy < 0) dy = 0;
                    if (dx < 0) dx = 0;
                    if (dy >= dataH) dy = dataH - 1;
                    if (dx >= dataW) dx = dataW - 1;
                    sum += h_Data[dy * dataW + dx] * h_Kernel[(kernelY - ky) * kernelW + (kernelX - kx)];
                }
                h_Result[y * dataW + x] = (float)sum;
            }
        }
    }
}

void modulateAndNormalize(
    fComplex* d_Dst,
    fComplex* d_DataSrc,
    fComplex* d_KernelSrc,
    int fftH,
    int fftW,
    int padding
) {
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);
    modulateAndNormalizeKernel << <iDivUp(dataSize, 256), 256 >> > (
        d_Dst,
        d_DataSrc,
        d_KernelSrc,
        dataSize,
        1.0f / (float)(fftW * fftH)
        );
    getLastCudaError("modulateAndNormalize() execution failed\n");
}
