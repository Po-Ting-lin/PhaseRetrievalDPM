#include <opencv2/opencv.hpp>

static void displayImage(const cv::Mat& image, const char* name, bool mag) {
	cv::Mat Out;
	if (mag) {
		cv::resize(image, Out, cv::Size(image.cols / 4, image.rows / 4), 5, 5);
	}
	else {
		image.copyTo(Out);
	}
	cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::moveWindow(name, 200, 200);
	cv::imshow(name, Out);
	cv::waitKey(0);
}

int main(void) {
	cv::Mat sp, bg;
	sp = cv::imread(R"(sp1.bmp)", cv::IMREAD_GRAYSCALE);
	bg = cv::imread(R"(bg1.bmp)", cv::IMREAD_GRAYSCALE);

	if (!sp.data || !bg.data) {
		std::cout << "Error: the image wasn't correctly loaded." << std::endl;
		return -1;
	}

	displayImage(sp, "sp1", true);
	displayImage(bg, "bg1", true);




	return 0;
}
