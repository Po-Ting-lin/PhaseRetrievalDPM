#include "phase_retriever.cuh"

#define DEBUG false
#define TIMER true

void PhaseRetriever(uchar* sp, uchar* bg, float* dst, float* spectrumDst, int width, int height, int spx, int spy, int bgx, int bgy) {
	PhaseRetrieverInfo info;
	info.Image = nullptr;
	info.WrappedImage = nullptr;
	info.UnwrappedImage = nullptr;
	info.Dst = dst;
	info.SpectrumDst = spectrumDst;
	info.Width = width;
	info.Height = height;
	info.CroppedWidth = width / 4;
	info.CroppedHeight = height / 4;
	info.CroppedSPOffsetX = spx;
	info.CroppedSPOffsetY = spy;
	info.CroppedBGOffsetX = bgx;
	info.CroppedBGOffsetY = bgy;
	info.NumberOfRealElements = width * height;
	info.NumberOfCropElements = (width / 4) * (height / 4);
	info.DataElementsPerStream = info.NumberOfRealElements / D_NUM_STREAMS;
	info.DataBytesPerStream = info.NumberOfRealElements * sizeof(uchar) / D_NUM_STREAMS;
	info.Blocks = new dim3(TILE_DIM, TILE_DIM);
	info.Blocks1D = new dim3(TILE_DIM * 2); // occupancy will be 100% for dim.x = 64
	info.Grids = new dim3(iDivUp(width, TILE_DIM), iDivUp(height, TILE_DIM));
	info.CroppedGrids = new dim3(iDivUp(width / 4, TILE_DIM), iDivUp(height / 4, TILE_DIM));
	info.Grids1D = new dim3(iDivUp(info.DataElementsPerStream, TILE_DIM * 2));
	gpuErrorCheck(cufftPlan2d(&info.fftHandle, info.Height, info.Width, CUFFT_C2C));
	gpuErrorCheck(cufftPlan2d(&info.ifftHandle, info.CroppedWidth, info.CroppedHeight, CUFFT_C2C));

	float* sp_unwarpped = nullptr;
	float* bg_unwarpped = nullptr;
	
#if TIMER
	auto t0 = std::chrono::system_clock::now();
#endif
	imageRetriever(sp, sp_unwarpped, info, true);
	imageRetriever(bg, bg_unwarpped, info, false);
	for (int i = 0; i < info.NumberOfCropElements; i++) {
		sp_unwarpped[i] -= bg_unwarpped[i];
	}
#if TIMER
	auto t1 = std::chrono::system_clock::now();
	printTime(t0, t1, "total elapsed time");
#endif
	
	free(bg_unwarpped);
	delete info.Blocks;
	delete info.Blocks1D;
	delete info.Grids;
	delete info.Grids1D;
	delete info.CroppedGrids;
}

void imageRetriever(uchar* src, float*& dst, PhaseRetrieverInfo& info, bool isSp) {
	info.Image = src;
#if TIMER
	auto t0 = std::chrono::system_clock::now();
#endif
	getWrappedImage(info, isSp);
#if TIMER
	auto t1 = std::chrono::system_clock::now();
#endif
	getUnwrappedImage(info, isSp);
#if TIMER
	auto t2 = std::chrono::system_clock::now();
	printTime(t0, t1, "getWrappedImage");
	printTime(t1, t2, "getUnwrappedImage");
#endif
	dst = info.UnwrappedImage;
}

void getWrappedImage(PhaseRetrieverInfo& info, bool isSp) {
	float* d_magnitude;
	float* d_firstCrop;
	uchar* image_ptr = info.Image;
	uchar* d_image_ptr;
	fComplex* d_data;
	fComplex* d_Spectrum;
	fComplex* d_SecondCrop;
	fComplex* d_rawWrapped;
	float* d_blk_vals;
	int* d_blk_idx;
	int* d_blk_num;
	int* d_max_index;
	int max_index = 0;
	const int block_size_1d = TILE_DIM * TILE_DIM;
	const int grid_size_1d = MIN(MAX_KERNEL_BLOCKS, ((info.NumberOfCropElements + block_size_1d - 1) / block_size_1d));
	const int max_block = (info.NumberOfCropElements / block_size_1d) + 1;

	// allocate the resources
	gpuErrorCheck(cudaMalloc((uchar**)&d_image_ptr, info.NumberOfRealElements * sizeof(uchar)));
	gpuErrorCheck(cudaMalloc((fComplex**)&d_data, info.NumberOfRealElements * sizeof(fComplex)));
	gpuErrorCheck(cudaMalloc((fComplex**)&d_Spectrum, info.NumberOfRealElements * sizeof(fComplex)));
	gpuErrorCheck(cudaMalloc((float**)&d_magnitude, info.NumberOfRealElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_firstCrop, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc(&d_blk_vals, max_block * sizeof(float)));
	gpuErrorCheck(cudaMalloc(&d_blk_idx, max_block * sizeof(int)));
	gpuErrorCheck(cudaMalloc(&d_blk_num, 1 * sizeof(int)));
	gpuErrorCheck(cudaMalloc(&d_max_index, sizeof(int)));
	gpuErrorCheck(cudaMalloc((fComplex**)&d_SecondCrop, info.NumberOfCropElements * sizeof(fComplex)));
	gpuErrorCheck(cudaMalloc((fComplex**)&d_rawWrapped, info.NumberOfCropElements * sizeof(fComplex)));
	gpuErrorCheck(cudaMalloc((float**)&info.WrappedImage, info.NumberOfCropElements * sizeof(float)));

	// aysn H to D
	int offset = 0;
	cudaStream_t stream[D_NUM_STREAMS];
	for (int i = 0; i < D_NUM_STREAMS; i++) {
		cudaStreamCreate(&stream[i]);
	}
	
	for (int i = 0; i < D_NUM_STREAMS; i++) {
		offset = i * info.DataElementsPerStream;
		gpuErrorCheck(cudaMemcpyAsync(&d_image_ptr[offset], &image_ptr[offset], info.DataBytesPerStream, cudaMemcpyHostToDevice, stream[i]));
		realToComplex << <*info.Grids1D, *info.Blocks1D, 0, stream[i] >> > (&d_image_ptr[offset], &d_data[offset], info.NumberOfRealElements);
	}
	gpuErrorCheck(cudaDeviceSynchronize());
	for (int i = 0; i < D_NUM_STREAMS; i++) {
		gpuErrorCheck(cudaStreamDestroy(stream[i]));
	}
	gpuErrorCheck(cufftExecC2C(info.fftHandle, (cufftComplex*)d_data, (cufftComplex*)d_Spectrum, CUFFT_FORWARD));
	gpuErrorCheck(cudaDeviceSynchronize());
	complexToMagnitude << <*info.Grids, *info.Blocks >> > (d_Spectrum, d_magnitude, info.Width, info.Height);
	gpuErrorCheck(cudaDeviceSynchronize());

#if DEBUG
	float* h_magnitude = (float*)malloc(info.NumberOfRealElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_magnitude, d_magnitude, info.NumberOfRealElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_magnitude, info.Width, info.Height, true);
#endif

	if (isSp) {
		gpuErrorCheck(cudaMemcpy(info.SpectrumDst, d_magnitude, info.NumberOfRealElements * sizeof(float), cudaMemcpyDeviceToHost));
	}

	// find max index
	copyInterferenceComponentRoughly << <*info.CroppedGrids, *info.Blocks >> > (d_magnitude, d_firstCrop, info.Width, info.Height, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

#if DEBUG
	float* h_firstCrop = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_firstCrop, d_firstCrop, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_firstCrop, info.CroppedWidth, info.CroppedHeight, true);
#endif

	cudaMemset(d_blk_vals, 0.0f, max_block * sizeof(float));
	cudaMemset(d_blk_idx, 0, max_block * sizeof(int));
	cudaMemset(d_blk_num, 0, 1 * sizeof(int));
	max_idx_kernel << <grid_size_1d, block_size_1d >> > (d_firstCrop, info.NumberOfCropElements, d_max_index, d_blk_vals, d_blk_idx, d_blk_num);
	cudaMemcpy(&max_index, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);

	int max_cropped_x = max_index % info.CroppedWidth;
	int max_cropped_y = max_index / info.CroppedWidth;
	int offset_x = isSp ? info.CroppedSPOffsetX : info.CroppedBGOffsetX;
	int offset_y = isSp ? info.CroppedSPOffsetY : info.CroppedBGOffsetY;
	int max_x = max_cropped_x < info.CroppedWidth / 2 ? max_cropped_x + 7 * info.Width / 8 : max_cropped_x - info.CroppedWidth / 2 + offset_x;
	int max_y = max_cropped_y + info.Height / 2 + offset_y;

#if DEBUG
	float* d_SecondCropDebug;
	gpuErrorCheck(cudaMalloc((float**)&d_SecondCropDebug, info.NumberOfCropElements * sizeof(float)));
	copyInterferenceComponentDebug << <*info.CroppedGrids, *info.Blocks >> > (d_magnitude, d_SecondCropDebug, max_x, max_y, info.Width, info.Height, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	float* h_SecondCropDebug = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_SecondCropDebug, d_SecondCropDebug, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_SecondCropDebug, info.CroppedWidth, info.CroppedHeight, false);
#endif
	copyInterferenceComponent << <*info.CroppedGrids, *info.Blocks >> > (d_Spectrum, d_SecondCrop, max_x, max_y, info.Width, info.Height, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	// ifft
	gpuErrorCheck(cufftExecC2C(info.ifftHandle, (cufftComplex*)d_SecondCrop, (cufftComplex*)d_rawWrapped, CUFFT_INVERSE));
	gpuErrorCheck(cudaDeviceSynchronize());

	// arctan
	applyArcTan << <*info.CroppedGrids, *info.Blocks >> > (d_rawWrapped, info.WrappedImage, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	// free
	gpuErrorCheck(cudaFree(d_blk_vals));
	gpuErrorCheck(cudaFree(d_blk_idx));
	gpuErrorCheck(cudaFree(d_blk_num));
	gpuErrorCheck(cudaFree(d_max_index));
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_magnitude));
	gpuErrorCheck(cudaFree(d_firstCrop));
	gpuErrorCheck(cudaFree(d_SecondCrop));
	gpuErrorCheck(cudaFree(d_rawWrapped));
}

void getUnwrappedImage(PhaseRetrieverInfo& info, bool isSp) {
#if DEBUG
	float* h_wrapped_image = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_wrapped_image, info.WrappedImage, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_wrapped_image, info.CroppedWidth, info.CroppedHeight, false);
	free(h_wrapped_image);
#endif
	float* h_sumC;
	float* d_dx;
	float* d_dy;
	float* d_sumC;
	float* d_divider;
	float tx = PI / info.CroppedWidth;
	float ty = PI / info.CroppedHeight;
	gpuErrorCheck(cudaMalloc((float**)&d_dx, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_dy, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_sumC, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_divider, info.NumberOfCropElements * sizeof(float)));

	// diff
	applyDifference<<<*info.CroppedGrids, *info.Blocks >>>(info.WrappedImage, d_dx, d_dy, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());
	applySum << <*info.CroppedGrids, *info.Blocks >> > (d_dx, d_dy, d_sumC, d_divider, tx, ty, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaFree(d_dx));
	gpuErrorCheck(cudaFree(d_dy));

	if (isSp) 
		h_sumC = info.Dst;
	else 
		h_sumC = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_sumC, d_sumC, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	float* h_divider = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_divider, d_divider, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));

	cv::Mat sumC_mat(info.CroppedHeight, info.CroppedWidth, CV_32FC1, h_sumC);
	cv::dct(sumC_mat, sumC_mat);
	for (int i = 0; i < info.NumberOfCropElements; i++) {
		if (h_divider[i] != 0.0f) h_sumC[i] = h_sumC[i] / h_divider[i];
	}
	cv::idct(sumC_mat, sumC_mat);
	info.UnwrappedImage = h_sumC;

	// free
	free(h_divider);
	gpuErrorCheck(cudaFree(d_sumC));
	gpuErrorCheck(cudaFree(d_divider));
	gpuErrorCheck(cudaFree(info.WrappedImage));
}
