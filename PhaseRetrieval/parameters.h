#pragma once
#define PI 3.141592654f
#define PI2 6.283185308f
#define TILE_DIM 32
#define D_NUM_STREAMS 4

#define MAX_KERNEL_BLOCKS 50
#define MIN(a,b) ((a>b)?b:a)
#define FLOAT_MIN -1.0f


#define ExportDll __declspec(dllexport)
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
    uchar* Image;
    float* WrappedImage;
    float* UnwrappedImage;
    float* Dst;
    float* SpectrumDst;
    int Width;
    int Height;
    int CroppedWidth;
    int CroppedHeight;
    int CroppedSPOffsetX;
    int CroppedBGOffsetX;
    int CroppedSPOffsetY;
    int CroppedBGOffsetY;
    int NumberOfRealElements;
    int NumberOfCropElements;
    int DataElementsPerStream;
    int DataBytesPerStream;
    int fftHandle;
    int ifftHandle;
    dim3* Grids;
    dim3* CroppedGrids;
    dim3* Blocks;
    dim3* Grids1D;
    dim3* Blocks1D;
};