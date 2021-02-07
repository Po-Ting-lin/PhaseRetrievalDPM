#include "phase_retriever.cuh"

int main(void) {
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

	auto t0 = std::chrono::system_clock::now();
	PhaseRetriever(sp.data, bg.data, result, sp.cols, sp.rows, spx, spy, bgx, bgy);
	auto t1 = std::chrono::system_clock::now();
	printTime(t0, t1, "total elapsed time");

	displayImage(result, 768, 768, false);

	return 0;
}
