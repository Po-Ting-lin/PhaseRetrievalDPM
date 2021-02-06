#include "phase_retriever.cuh"
#include <cufft.h>
#include <iostream>

#define D_NUM_STREAMS 8
#define DEBUG false
#define TIMER true

void phaseRetriever(cv::Mat& sp, cv::Mat& bg, float*& dst) {
	PhaseRetrieverInfo info;
	info.Image = nullptr;
	info.WrappedImage = nullptr;
	info.UnwrappedImage = nullptr;
	info.Width = sp.cols;
	info.Height = sp.rows;
	info.CroppedWidth = sp.cols / 4;
	info.CroppedHeight = sp.rows / 4;
	info.NumberOfRealElements = sp.cols * sp.rows;
	info.NumberOfCropElements = (sp.cols / 4) * (sp.rows / 4);
	info.Blocks = new dim3(TILE_DIM, TILE_DIM);
	info.Grids = new dim3(iDivUp(sp.cols, TILE_DIM), iDivUp(sp.rows, TILE_DIM));
	info.CroppedGrids = new dim3(iDivUp(sp.cols / 4, TILE_DIM), iDivUp(sp.rows / 4, TILE_DIM));
	
	float* sp_unwarpped = nullptr;
	float* bg_unwarpped = nullptr;
	processPhaseRetriever(sp, sp_unwarpped, info);
	processPhaseRetriever(bg, bg_unwarpped, info);

	for (int i = 0; i < info.NumberOfCropElements; i++) {
		sp_unwarpped[i] -= bg_unwarpped[i];
	}
	dst = sp_unwarpped;
	
	free(bg_unwarpped);
	delete info.Blocks;
	delete info.Grids;
	delete info.CroppedGrids;
}

void processPhaseRetriever(cv::Mat& src, float*& dst, PhaseRetrieverInfo& info) {
	info.Image = &src;

#if false
	auto t0 = std::chrono::system_clock::now();
#endif
	
	getWrappedImage(info);

#if false
	auto t1 = std::chrono::system_clock::now();
#endif
	getUnwrappedImage(info);

#if false
	auto t2 = std::chrono::system_clock::now();
	printTime(t0, t1, "getWrappedImage");
	printTime(t1, t2, "getUnwrappedImage");
#endif
	dst = info.UnwrappedImage;
}

void getWrappedImage(PhaseRetrieverInfo& info) {
	float* d_magnitude;
	float* d_firstCrop;
	uchar* image_ptr = info.Image->data;
	uchar* d_image_ptr;
	fComplex* d_data;
	fComplex* d_Spectrum;
	fComplex* d_SecondCrop;
	fComplex* d_rawWrapped;

	// make a FFT plan
	cufftHandle fftPlan;
	cufftHandle ifftPlan;
	gpuErrorCheck(cufftPlan2d(&fftPlan, info.Height, info.Width, CUFFT_C2C));
	gpuErrorCheck(cufftPlan2d(&ifftPlan, info.CroppedWidth, info.CroppedHeight, CUFFT_C2C));

	// aysn H to D
	cudaStream_t stream[D_NUM_STREAMS];
	for (int i = 0; i < D_NUM_STREAMS; i++) {
		cudaStreamCreate(&stream[i]);
	}
	gpuErrorCheck(cudaMalloc((uchar**)&d_image_ptr, info.NumberOfRealElements * sizeof(uchar)));
	gpuErrorCheck(cudaMalloc((fComplex**)&d_data, info.NumberOfRealElements * sizeof(fComplex)));
	int offset = 0;
	int data_elements_per_stream = info.NumberOfRealElements / D_NUM_STREAMS;
	int data_bytes_per_stream = info.NumberOfRealElements * sizeof(uchar) / D_NUM_STREAMS;
	dim3 grid(iDivUp(data_elements_per_stream, TILE_DIM));
	dim3 block(TILE_DIM);
	for (int i = 0; i < D_NUM_STREAMS; i++) {
		offset = i * data_elements_per_stream;
		gpuErrorCheck(cudaMemcpyAsync(&d_image_ptr[offset], &image_ptr[offset], data_bytes_per_stream, cudaMemcpyHostToDevice, stream[i]));
		realToComplex << <grid, block, 0, stream[i] >> > (&d_image_ptr[offset], &d_data[offset], info.NumberOfRealElements);
	}
	gpuErrorCheck(cudaDeviceSynchronize());
	for (int i = 0; i < D_NUM_STREAMS; i++) {
		gpuErrorCheck(cudaStreamDestroy(stream[i]));
	}

	// fft
	gpuErrorCheck(cudaMalloc((fComplex**)&d_Spectrum, info.NumberOfRealElements * sizeof(fComplex)));
	gpuErrorCheck(cufftExecC2C(fftPlan, (cufftComplex*)d_data, (cufftComplex*)d_Spectrum, CUFFT_FORWARD));
	gpuErrorCheck(cudaDeviceSynchronize());

	// magnitude
	gpuErrorCheck(cudaMalloc((float**)&d_magnitude, info.NumberOfRealElements * sizeof(float)));
	complexToMagnitude << <*info.Grids, *info.Blocks >> > (d_Spectrum, d_magnitude, info.Width, info.Height);
	gpuErrorCheck(cudaDeviceSynchronize());

#if DEBUG
	float* h_magnitude = (float*)malloc(info.NumberOfRealElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_magnitude, d_magnitude, info.NumberOfRealElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_magnitude, info.Width, info.Height, true);
#endif

	// find max index
	gpuErrorCheck(cudaMalloc((float**)&d_firstCrop, info.NumberOfCropElements * sizeof(float)));
	copyInterferenceComponentRoughly << <*info.CroppedGrids, *info.Blocks >> > (d_magnitude, d_firstCrop, info.Width, info.Height, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

#if DEBUG
	float* h_firstCrop = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_firstCrop, d_firstCrop, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_firstCrop, info.CroppedWidth, info.CroppedHeight, true);
#endif

	thrust::device_ptr<float> d_ptr(d_firstCrop);
	thrust::device_vector<float> d_vec(d_ptr, d_ptr + info.NumberOfCropElements);
	thrust::device_vector<float>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());
	unsigned int index = iter - d_vec.begin();
	float max_val = *iter;
	int max_cropped_x = index % info.CroppedWidth;
	int max_cropped_y = index / info.CroppedWidth;
	int max_x = max_cropped_x < info.CroppedWidth / 2 ? max_cropped_x + 7 * info.Width / 8 : max_cropped_x - info.CroppedWidth / 2;
	int max_y = max_cropped_y + info.Height / 2;
	//std::cout << "Position is x: " << max_loc_x << "y: " << max_loc_y << std::endl;

	// crop
#if DEBUG
	float* d_SecondCropDebug;
	gpuErrorCheck(cudaMalloc((float**)&d_SecondCropDebug, info.NumberOfCropElements * sizeof(float)));
	copyInterferenceComponentDebug << <*info.CroppedGrids, *info.Blocks >> > (d_magnitude, d_SecondCropDebug, max_x, max_y, info.Width, info.Height, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	float* h_SecondCropDebug = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_SecondCropDebug, d_SecondCropDebug, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_SecondCropDebug, info.CroppedWidth, info.CroppedHeight, false);
#endif
	gpuErrorCheck(cudaMalloc((fComplex**)&d_SecondCrop, info.NumberOfCropElements * sizeof(fComplex)));
	copyInterferenceComponent << <*info.CroppedGrids, *info.Blocks >> > (d_Spectrum, d_SecondCrop, max_x, max_y, info.Width, info.Height, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	// ifft
	gpuErrorCheck(cudaMalloc((fComplex**)&d_rawWrapped, info.NumberOfCropElements * sizeof(fComplex)));
	gpuErrorCheck(cufftExecC2C(ifftPlan, (cufftComplex*)d_SecondCrop, (cufftComplex*)d_rawWrapped, CUFFT_INVERSE));
	gpuErrorCheck(cudaDeviceSynchronize());

	// arctan
	gpuErrorCheck(cudaMalloc((float**)&info.WrappedImage, info.NumberOfCropElements * sizeof(float)));
	applyArcTan << <*info.CroppedGrids, *info.Blocks >> > (d_rawWrapped, info.WrappedImage, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	// free
	gpuErrorCheck(cufftDestroy(fftPlan));
	gpuErrorCheck(cufftDestroy(ifftPlan));
	gpuErrorCheck(cudaFree(d_data));
	gpuErrorCheck(cudaFree(d_magnitude));
	gpuErrorCheck(cudaFree(d_firstCrop));
	gpuErrorCheck(cudaFree(d_SecondCrop));
	gpuErrorCheck(cudaFree(d_rawWrapped));
}

void getUnwrappedImage(PhaseRetrieverInfo& info) {
#if DEBUG
	float* h_wrapped_image = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_wrapped_image, info.WrappedImage, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_wrapped_image, info.CroppedWidth, info.CroppedHeight, false);
	free(h_wrapped_image);
#endif

	float* d_dx;
	float* d_dy;
	float* d_sumC;
	float* d_divider;
	gpuErrorCheck(cudaMalloc((float**)&d_dx, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_dy, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_sumC, info.NumberOfCropElements * sizeof(float)));
	gpuErrorCheck(cudaMalloc((float**)&d_divider, info.NumberOfCropElements * sizeof(float)));
	float tx = PI / info.CroppedWidth;
	float ty = PI / info.CroppedHeight;
	// diff
	applyDifference<<<*info.CroppedGrids, *info.Blocks >>>(info.WrappedImage, d_dx, d_dy, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	
#if DEBUG
	float* h_dx = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_dx, d_dx, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	float* h_dy = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_dy, d_dy, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	//displayImage(h_dx, info.CroppedWidth, info.CroppedHeight, false);

	std::cout << "***********************dx section***********************" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << "i: " << i << "; value: " << h_dx[i] << std::endl;
	}
	for (int i = info.CroppedWidth - 10; i < info.CroppedWidth; i++) {
		std::cout << "i: " << i << "; value: " << h_dx[i] << std::endl;
	}
	std::cout << "***********************dy section***********************" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << "i: " << i << "; value: " << h_dy[i] << std::endl;
	}
	for (int i = info.CroppedWidth - 10; i < info.CroppedWidth; i++) {
		std::cout << "i: " << i << "; value: " << h_dy[i] << std::endl;
	}
	free(h_dx);
	free(h_dy);
#endif
	
	applySum << <*info.CroppedGrids, *info.Blocks >> > (d_dx, d_dy, d_sumC, d_divider, tx, ty, info.CroppedWidth, info.CroppedHeight);
	gpuErrorCheck(cudaDeviceSynchronize());

	gpuErrorCheck(cudaFree(d_dx));
	gpuErrorCheck(cudaFree(d_dy));

	float* h_sumC = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_sumC, d_sumC, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));
	float* h_divider = (float*)malloc(info.NumberOfCropElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_divider, d_divider, info.NumberOfCropElements * sizeof(float), cudaMemcpyDeviceToHost));

#if DEBUG
	std::cout << "***********************sumC section***********************" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << "i: " << i << "; value: " << h_sumC[i] << std::endl;
	}
	for (int i = info.CroppedWidth - 10; i < info.CroppedWidth; i++) {
		std::cout << "i: " << i << "; value: " << h_sumC[i] << std::endl;
	}
	std::cout << "***********************divider section***********************" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << "i: " << i << "; value: " << h_divider[i] << std::endl;
	}
	for (int i = info.CroppedWidth - 10; i < info.CroppedWidth; i++) {
		std::cout << "i: " << i << "; value: " << h_divider[i] << std::endl;
	}
#endif
	cv::Mat sumC_mat(info.CroppedHeight, info.CroppedWidth, CV_32FC1, h_sumC);
	cv::dct(sumC_mat, sumC_mat, cv::DCT_ROWS);
	cv::transpose(sumC_mat, sumC_mat);
	cv::dct(sumC_mat, sumC_mat, cv::DCT_ROWS);
	cv::transpose(sumC_mat, sumC_mat);
	
#if DEBUG
	for (int i = 0; i < 10; i++) {
		std::cout << "i: " << i << "; value: " << h_sumC[i] << std::endl;
	}
	for (int i = 0 + info.CroppedWidth; i < 10 + info.CroppedWidth; i++) {
		std::cout << "i: " << i << "; value: " << h_sumC[i] << std::endl;
	}
#endif
	
	// divide, can be optimized
	for (int i = 0; i < info.NumberOfCropElements; i++) {
		if (h_divider[i] != 0.0f) {
			h_sumC[i] = h_sumC[i] / h_divider[i];
		}
	}

	// iDCT
	cv::idct(sumC_mat, sumC_mat, cv::DCT_ROWS);
	cv::transpose(sumC_mat, sumC_mat);
	cv::idct(sumC_mat, sumC_mat, cv::DCT_ROWS);
	cv::transpose(sumC_mat, sumC_mat);

#if DEBUG
	for (int i = 0; i < 10; i++) {
		std::cout << "i: " << i << "; value: " << h_sumC[i] << std::endl;
	}
	for (int i = 0 + info.CroppedWidth; i < 10 + info.CroppedWidth; i++) {
		std::cout << "i: " << i << "; value: " << h_sumC[i] << std::endl;
	}
	displayImage(h_sumC, info.CroppedWidth, info.CroppedHeight, false);
#endif

	info.UnwrappedImage = h_sumC;

	// free
	free(h_divider);
	gpuErrorCheck(cudaFree(d_sumC));
	gpuErrorCheck(cudaFree(d_divider));
	gpuErrorCheck(cudaFree(info.WrappedImage));
}
