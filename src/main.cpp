#include "canny.h"

int main(){
	Mat I;

	I = imread("../road.jpg");

	imshow("Original",I);

	canny ced(I);

	ced.denoising();
	ced.get_gradient();
	ced.get_angles();
	ced.non_maximum_supression();

	//Calculate the mean value of the gradient after non-maximum supression
	//Use this value as the high threshold
	//And use half of it as the low threshold
	Scalar mean_value = mean(ced.G, ced.supression_mask);
	float high_threshold = mean_value[0];
	float low_threshold = high_threshold/2;

	ced.get_strong_edges(high_threshold);
	ced.get_weak_edges(low_threshold, high_threshold);

	imshow("Strong edges",ced.strong_edges);
	imshow("Weak edges",ced.weak_edges);

	ced.blob_analysis();

	imshow("Canny result",ced.canny_result);

	waitKey();
	
	return 0;
}