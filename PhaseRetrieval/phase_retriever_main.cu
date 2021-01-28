#include "phase_retriever.cuh"

PhaseRetriever::PhaseRetriever(cv::Mat& src) {
	_image = src.data;
	_width = src.cols;
	_height = src.rows;
}

PhaseRetriever::~PhaseRetriever() {
	
}

void PhaseRetriever::Process() {

}
