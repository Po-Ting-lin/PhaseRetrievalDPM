#pragma once
#define PI 3.141592654f
#define PI2 6.283185308f
#define TILE_DIM 32
#define D_NUM_STREAMS 8
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
    dim3* Grids;
    dim3* CroppedGrids;
    dim3* Blocks;
    dim3* Grids1D;
    dim3* Blocks1D;
};