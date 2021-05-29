#include<windows.h>
#include "phase_retriever.cuh"
#include "warmup.cuh"

int main(void) {
	warmUpGpu();
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	cv::Mat sp, bg;
	sp = cv::imread(R"(sp1.bmp)", cv::IMREAD_GRAYSCALE);
	bg = cv::imread(R"(bg1.bmp)", cv::IMREAD_GRAYSCALE);

	if (!sp.data || !bg.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}
	float* result = new float[768 * 768];
	int spx = 0;
	int spy = 0;
	int bgx = 0;
	int bgy = 0;
	PhaseRetriever(sp.data, bg.data, result, sp.cols, sp.rows, spx, spy, bgx, bgy);
	displayImage(result, 768, 768, false);

	return 0;
}
