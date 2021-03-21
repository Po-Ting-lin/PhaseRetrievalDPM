#include "warmup.cuh"

__global__ void warmUpKernel() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float ia, ib;
	ia = ib = 0.0f;
	ib += ia + tid;
}

void warmUpGpu() {
	dim3 block(32, 32);
	dim3 grid(96, 96);
	warmUpKernel << <grid, block >> > ();
	cudaDeviceSynchronize();
}