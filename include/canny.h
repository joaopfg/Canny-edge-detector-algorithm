#ifndef CANNY_H_
#define CANNY_H_

#include <bits/stdc++.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class canny{
public:
	//<<I_gray>> will maintain the image converted to gray scale
	//<<Gx>> will maintain the x-component of the gradient
	//<<Gy>> will maintain the y-component of the gradient
	//<<G>> will maintain the gradient
	//<<Angle>> will maintain the angle between <<G>> and <<Gx>>
	//<<G_supressed>> will only maintain those pixels of <<G>> which passed the non-maximum supression test
	//<<supression_mask>> will be used as an auxiliar to calculate the mean of all good pixels in <<G_supressed>> 
	//<<G_candidates>> will maintain only the pixels in <<G_supressed>> already obtained in <<G_strong>> and the pixels with gradient between the low and the high threshold
	//<<visit>> will be an auxiliary matrix in the search for connected components
	//<<strong_edges>> will maintain a white pixel for the pixels that are not black in <<G_strong>> and a black pixel for the others
	//<<candidates>> will store the position of the pixels in <<strong_edges>>
	//<<weak_edges>> will maintain a white pixel for the pixels that are not black in G_candidates and a black pixel for the others
	//<<canny_vector>> will maintain the position of the pixels in <<G_candidates>> which are directly connected to a connected component in <<G_strong>> as well as the pixels in the connected componets of <<G_strong>>
	//<<canny_result>> will maintain a image with the pixels stored in <<canny_vector>> as white pixels and the others as black pixels. It's the final result
	
	Mat I_gray, Gx, Gy, G, Angle, G_supressed, supression_mask, G_candidates, visit, strong_edges, weak_edges, canny_result;
	vector< pair<int,int> > candidates;  
	vector< pair<int,int> > canny_vector;

	//Constructor
	canny(const Mat& I);

	//Pass a gaussian filter through the image
	void denoising();

	//Pass a sobel filter through the image
	void get_gradient();

	//fill the <<Angle>> matrix
	//angle = 0 will mean that the gradient vector is near the x-axis
	//angle = 1 will mean that the gradient vector is near the y-axis
	//angle = 3 will mean that the gradient vector is near the y = x line
	//angle = 4 will mean that the gradient vector is near the y = -x line
	void get_angles();

	//Compare the pixel intensity with the intensities of the pixels in the rounded direction
	//Maintain only those pixels with intensity greater than the two nearest pixels in the rounded direction 
	void non_maximum_supression();

	//fill <<candidates>> and <<strong edges>>
	void get_strong_edges(float high_threshold);

	//fill <<G_candidates>> and <<weak_edges>>
	void get_weak_edges(float low_threshold, float high_threshold);

	//Do a depth first search starting from point p in order to fill <<canny_vector>>
	void dfs(pair<int,int> p);

	//Do a depth first search starting from each point that are still not visited in <<candidates>>
	void blob_analysis();
};

#endif