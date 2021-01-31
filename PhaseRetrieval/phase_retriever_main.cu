#include "phase_retriever.cuh"
#include <cufft.h>

#define DEBUG false

void processPhaseRetriever(cv::Mat& src) {
	PhaseRetrieverInfo info;
	info.Image = &src;
	info.Width = src.cols;
	info.HalfWidth = (src.cols / 2 + 1);
	info.Height = src.rows;
	info.NumberOfRealElements = src.cols * src.rows;
	info.NumberOfComplexElements = (src.cols / 2 + 1) * src.rows;
	getWrappedImage(info);
	getUnwrappedImage();
}

void getWrappedImage(PhaseRetrieverInfo& info) {
	// fft
	uchar* image_ptr = info.Image->data;
	float* d_data;
	fComplex* d_Spectrum;

	float* h_data = new float[info.Width * info.Height];
	for (int i = 0; i < info.Width * info.Height; i++) {
		h_data[i] = image_ptr[i];
	}

	cufftHandle fftPlan;
	long src_byte_size = info.NumberOfRealElements * sizeof(float);
	gpuErrorCheck(cudaMalloc((float**)&d_data, src_byte_size));
	gpuErrorCheck(cudaMalloc((void**)&d_Spectrum, info.NumberOfComplexElements * sizeof(fComplex)));
	gpuErrorCheck(cudaMemcpy(d_data, h_data, src_byte_size, cudaMemcpyHostToDevice));
	gpuErrorCheck(cufftPlan2d(&fftPlan, info.Height, info.Width, CUFFT_R2C));
	gpuErrorCheck(cufftExecR2C(fftPlan, (cufftReal*)d_data, (cufftComplex*)d_Spectrum));
	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaFree(d_data));


	// magnitude

	float* d_magnitude;
	gpuErrorCheck(cudaMalloc((float**)&d_magnitude, info.NumberOfComplexElements * sizeof(float)));
	dim3 blocks(TILE_DIM, TILE_DIM);
	dim3 grids(iDivUp(info.HalfWidth, TILE_DIM), iDivUp(info.Height, TILE_DIM));
	complexToMagnitude << <grids, blocks >> > (d_Spectrum, d_magnitude, info.HalfWidth, info.Height);
	gpuErrorCheck(cudaDeviceSynchronize());

#if DEBUG
	float* h_magnitude = (float*)malloc(info.NumberOfComplexElements * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_magnitude, d_magnitude, info.NumberOfComplexElements * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_magnitude, info.HalfWidth, info.Height, true);
#endif

	// find max index
	float* d_firstCrop;
	int frist_cropped_width = info.Width / 8;
	int first_cropped_height = info.Height / 2;
	gpuErrorCheck(cudaMalloc((float**)&d_firstCrop, frist_cropped_width * first_cropped_height * sizeof(float)));
	copyInterferenceComponentRoughly << <grids, blocks >> > (d_magnitude, d_firstCrop, info.Height / 4, frist_cropped_width, first_cropped_height, info.HalfWidth, info.Height, frist_cropped_width);
	gpuErrorCheck(cudaDeviceSynchronize());

#if DEBUG
	float* h_firstCrop = (float*)malloc(first_cropped_height * frist_cropped_width * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_firstCrop, d_firstCrop, first_cropped_height * frist_cropped_width * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_firstCrop, frist_cropped_width, first_cropped_height, true);
#endif

	thrust::device_ptr<float> d_ptr(d_firstCrop);
	thrust::device_vector<float> d_vec(d_ptr, d_ptr + frist_cropped_width * first_cropped_height);
	thrust::device_vector<float>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());
	unsigned int index = iter - d_vec.begin();
	float max_val = *iter;
	int max_loc_x = index % frist_cropped_width;
	int max_loc_y = index / frist_cropped_width + info.Height / 4;
	std::cout << "Position is x: " << max_loc_x << "y: "<< max_loc_y << std::endl;
	gpuErrorCheck(cudaFree(d_firstCrop));


	// crop
	int second_cropped_width = info.Width / 4;
	int second_cropped_height = info.Height / 4;
	dim3 croppedGrids(iDivUp(second_cropped_width, TILE_DIM), iDivUp(second_cropped_height, TILE_DIM));

#if DEBUG
	float* d_SecondCropDebug;
	gpuErrorCheck(cudaMalloc((float**)&d_SecondCropDebug, second_cropped_width * second_cropped_height * sizeof(float)));
	copyInterferenceComponentDebug << <croppedGrids, blocks >> > (d_magnitude, d_SecondCropDebug, max_loc_x, max_loc_y, info.HalfWidth, info.Height, second_cropped_width, second_cropped_height);
	gpuErrorCheck(cudaDeviceSynchronize());

	float* h_SecondCropDebug = (float*)malloc(second_cropped_width * second_cropped_height * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_SecondCropDebug, d_SecondCropDebug, second_cropped_height * second_cropped_width * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_SecondCropDebug, second_cropped_width, second_cropped_height, false);
#endif
	
	fComplex* d_SecondCrop;
	gpuErrorCheck(cudaMalloc((fComplex**)&d_SecondCrop, second_cropped_width * second_cropped_height * sizeof(fComplex)));
	copyInterferenceComponent << <croppedGrids, blocks >> > (d_Spectrum, d_SecondCrop, max_loc_x, max_loc_y, info.HalfWidth, info.Height, second_cropped_width, second_cropped_height);
	gpuErrorCheck(cudaDeviceSynchronize());

	// ifft
	fComplex* d_rawWrapped;
	cufftHandle ifftPlan;
	gpuErrorCheck(cudaMalloc((fComplex**)&d_rawWrapped, second_cropped_width * second_cropped_height * sizeof(fComplex)));
	gpuErrorCheck(cufftPlan2d(&ifftPlan, second_cropped_width, second_cropped_height, CUFFT_C2C));
	gpuErrorCheck(cufftExecC2C(ifftPlan, (cufftComplex*)d_SecondCrop, (cufftComplex*)d_rawWrapped, CUFFT_INVERSE));
	gpuErrorCheck(cudaDeviceSynchronize());
	gpuErrorCheck(cudaFree(d_SecondCrop));

	// test ifft
	//float* d_rawV;
	//gpuErrorCheck(cudaMalloc((float**)&d_rawV, second_cropped_width * second_cropped_height * sizeof(float)));
	//complexToMagnitude << <croppedGrids, blocks >> > (d_rawWrapped, d_rawV, second_cropped_width, second_cropped_height);
	//gpuErrorCheck(cudaDeviceSynchronize());
	//float* h_rawWrapped = (float*)malloc(second_cropped_width * second_cropped_height * sizeof(float));
	//gpuErrorCheck(cudaMemcpy(h_rawWrapped, d_rawV, second_cropped_height * second_cropped_width * sizeof(float), cudaMemcpyDeviceToHost));
	//displayImage(h_rawWrapped, second_cropped_width, second_cropped_height, false);

	//// arctan
	float* d_wrapped;
	gpuErrorCheck(cudaMalloc((float**)&d_wrapped, second_cropped_width * second_cropped_height * sizeof(float)));
	applyArcTan << <croppedGrids, blocks >> > (d_rawWrapped, d_wrapped, second_cropped_width, second_cropped_height);
	gpuErrorCheck(cudaDeviceSynchronize());

	float* h_wrapped = (float*)malloc(second_cropped_width * second_cropped_height * sizeof(float));
	gpuErrorCheck(cudaMemcpy(h_wrapped, d_wrapped, second_cropped_height * second_cropped_width * sizeof(float), cudaMemcpyDeviceToHost));
	displayImage(h_wrapped, second_cropped_width, second_cropped_height, false);
}

void getUnwrappedImage() {

}



//////
//#define IMAGE_DIM 256
//
//grid = dim3(IMAGE_DIM / 16, IMAGE_DIM / 16, 1);
//
//threads = dim3(16, 16, 1);
//
//// Declare handles to the FFT plans
//
//cufftHandle forwardFFTPlan;
//
//cufftHandle inverseFFTPlan;
//
//// Create the plans -- forward and reverse (Real2Complex, Complex2Real)
//
//CUFFT_SAFE_CALL(cufftPlan2d(&forwardFFTPlan, IMAGE_DIM, IMAGE_DIM, CUFFT_R2C));
//
//CUFFT_SAFE_CALL(cufftPlan2d(&inverseFFTPlan, IMAGE_DIM, IMAGE_DIM, CUFFT_C2R));
//
//int num_real_elements = IMAGE_DIM * IMAGE_DIM;
//
//int num_complex_elements = IMAGE_DIM * (IMAGE_DIM / 2 + 1);
//
//// HOST MEMORY
//
//float* h_img;
//
//float* h_imgF;
//
//// ALLOCATE HOST MEMORY
//
//h_img = (float*)malloc(m_num_real_elements * sizeof(float));
//
//h_complex_imgSpec = (cufftComplex*)malloc(m_num_complex_elements * sizeof(cufftComplex));
//
//h_imgF = (float*)malloc(m_num_real_elements * sizeof(float));
//
//for (int x = 0; x < IMAGE_DIM; x++)
//
//{
//
//	for (int y = 0; y < IMAGE_DIM; y++)
//
//	{
//
//		// initialize the input image memory somehow
//
//		// (this probably comes from a file or image buffer or something)
//
//		h_img[y * IMAGE_DIM + x] = 0.0f;
//
//	}
//
//}
//
//// DEVICE MEMORY
//
//float* d_img;
//
//cufftComplex* d_complex_imgSpec;
//
//float* d_imgF;
//
//// ALLOCATE DEVICE MEMORY
//
//(cudaMalloc((void**)&img, m_num_real_elements * sizeof(float)));
//
//(cudaMalloc((void**)&d_complex_imgSpec, m_num_complex_elements * sizeof(cufftComplex)));
//
//(cudaMalloc((void**)&img, m_num_real_elements * sizeof(float)));
//
//// copy host memory to device (input image)
//
//(cudaMemcpy(d_img, h_img, m_num_real_elements * sizeof(float), cudaMemcpyHostToDevice));
//
//
//
//// now run the forward FFT on the device (real to complex)
//
//CUFFT_SAFE_CALL(cufftExecR2C(forwardFFTPlan, d_img, d_complex_imgSpec));
//
//// copy the DEVICE complex data to the HOST
//
//// NOTE: we are only doing this so that you can see the data -- in general you want
//
//// to do your computation on the GPU without wasting the time of copying it back to the host
//
//(cudaMemcpy(h_complex_imgSpec, d_complex_imgSpec, m_num_complex_elements * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
//
//// print the complex data so you can see what it looks like
//
//for (int x = 0; x < (IMAGE_DIM / 2 + 1); x++)
//
//{
//
//	for (int y = 0; y < IMAGE_DIM; y++)
//
//	{
//
//		// initialize the input image memory somehow
//
//		// (this probably comes from a file or image buffer or something)
//
//		printf("h_complex_imgSpec[%d,%d] = %f + %fi\n", x, y, h_complex_imgSpec[y * (IMAGE_DIM / 2 + 1) + x].x, h_complex_imgSpec[y * (IMAGE_DIM / 2 + 1) + x].y);
//
//	}
//
//}
//
//// here you can modify or filter the data in the frequency domain
//
//// TODO: insert your filter code here, or whatever
//
//// NOTE: you can/should modify it on the GPU/DEVICE, not the HOST
//
//// IF you modify it on the HOST you will need to cudaMemcpy it back to the DEVICE
//
//// now run the inverse FFT on the device (complex to real)
//
//cufftExecC2R(inverseFFTPlan, d_complex_imgSpec, d_imgF);
//
//// NOTE: the data in d_imgF is not normalized at this point
//
//// Normalize the data in place - IFFT Normalization is 
//
//// dividing all elements by the total numbers of elements in the matrix/image/array (ie, number of pixels)
//
//NormalizeIFFT << < grid, threads >> > (d_imgF, IMAGE_DIM, IMAGE_DIM, 256.0f * 256.0f);
//
//// Copy the DEVICE memory to the HOST memory
//
//(cudaMemcpy(h_imgF, d_imgF, m_num_real_elements * sizeof(float), cudaMemcpyDeviceToHost));
//
//// print the elements of the resulting data
//
//for (int i = 0; i < m_num_real_elements; i++)
//
//{
//
//	printf("h_imgF[%d] = %f\n", i, h_imgF[i]);
//
//}
//
//// CLEANUP HOST MEMORY
//
//free(h_img);
//
//free(h_imgF);
//
//// CLEANUP DEVICE MEMORY
//
//(cudaFree(d_img));
//
//(cudaFree(d_complex_imgSpec));
//
//(cudaFree(d_imgF));