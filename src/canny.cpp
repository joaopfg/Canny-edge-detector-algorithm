#include "canny.h"

const int d[] = {0,-1,1};

canny::canny(const Mat& I){
	cvtColor(I, I_gray, COLOR_BGR2GRAY);
	Gx = Mat::zeros(I.rows, I.cols, CV_16SC1);
	Gy = Mat::zeros(I.rows, I.cols, CV_16SC1);
	G = Mat::zeros(I.rows, I.cols, CV_32FC1);
	Angle = Mat::zeros(I.rows, I.cols, CV_8UC1);
	G_supressed = Mat::zeros(I.rows, I.cols, CV_32FC1);
	supression_mask = Mat::zeros(I.rows, I.cols, CV_8UC1);
	G_candidates = Mat::zeros(I.rows, I.cols, CV_32FC1);
	visit = Mat::zeros(I.rows, I.cols, CV_8UC1);
	strong_edges = Mat::zeros(I.rows, I.cols, CV_8UC1);
	weak_edges = Mat::zeros(I.rows, I.cols, CV_8UC1);
	canny_result = Mat::zeros(I.rows, I.cols, CV_8UC1);
}

void canny::denoising(){
	float GaussianKernel[5][5] = {{2 , 4 , 5 , 4 , 2}, {4 , 9 , 12 , 9 , 4}, {5 , 12 , 15 ,12 , 5}, {4 , 9 , 12 , 9 , 4}, {2 , 4 , 5 , 4 , 2}};
	vector<float> GaussianBlur;

	for(int i=0;i<I_gray.rows - 5;i++){
		for(int j=0;j<I_gray.cols - 5;j++){
			Mat window_image(I_gray, Rect(j, i, 5, 5));

			for(int k=0;k<window_image.rows;k++){
				for(int m=0;m<window_image.cols;m++){
					GaussianBlur.push_back(window_image.at<uchar>(k,m)*GaussianKernel[k][m]); 
				}
			}

			for(int k=0;k<GaussianBlur.size();k++){
				if(k == 12) continue;

				GaussianBlur[12] += GaussianBlur[k];
			}

			I_gray.at<uchar>(i+2,j+2) = (uchar)(GaussianBlur[12] / 159);
			GaussianBlur.clear();
		}
	}
}

void canny::get_gradient(){
	int GxKernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
	int GyKernel[3][3] = {{-1, -2 ,-1}, {0, 0, 0}, {1, 2, 1}};

	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			Mat window_imageX(I_gray, Rect(j, i, 3, 3));
			Mat window_imageY(I_gray, Rect(j, i, 3, 3));

			for(int k=0;k<window_imageX.rows;k++){
				for(int m=0;m<window_imageX.cols;m++){
					window_imageX.at<int>(k,m) *= GxKernel[k][m];
					window_imageY.at<int>(k,m) *= GyKernel[k][m];
				}
			}

			for(int k=0;k<window_imageX.rows;k++){
				for(int m=0;m<window_imageX.cols;m++){
					if(k == 1 && m == 1) continue;

					window_imageX.at<int>(1,1) += window_imageX.at<int>(k,m);
					window_imageY.at<int>(1,1) += window_imageY.at<int>(k,m);
				}
			}

			Gx.at<int>(i+1,j+1) = window_imageX.at<int>(1,1);
			Gy.at<int>(i+1,j+1) = window_imageY.at<int>(1,1);
			G.at<float>(i+1,j+1) = hypot(Gx.at<int>(i+1,j+1), Gy.at<int>(i+1,j+1));
		}
	}
}

void canny::get_angles(){
	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			float ang = atan2(1.0*Gy.at<int>(i,j), 1.0*Gx.at<int>(i,j));
			float PI = acos(-1.0);
			
			if(ang >= 0.0 && ang < PI/4.0){
				if(ang < (PI/4.0) - ang) Angle.at<uchar>(i+1,j+1) = (uchar)0;
				else Angle.at<uchar>(i+1,j+1) = (uchar)3;
			}
			else if(ang >= PI/4.0 && ang < PI/2.0){
				if(ang - (PI/4.0) < (PI/2.0) - ang) Angle.at<uchar>(i+1,j+1) = (uchar)3;
				else Angle.at<uchar>(i+1,j+1) = (uchar)1; 
			}
			else if(ang >= PI/2.0 && ang < (3.0*PI)/4.0){
				if(ang - (PI/2.0) < ((3.0*PI)/4.0) - ang) Angle.at<uchar>(i+1,j+1) = (uchar)1;
				else Angle.at<uchar>(i+1,j+1) = (uchar)4;
			}
			else if(ang >= (3.0*PI)/4.0 && ang < PI){
				if(ang - ((3.0*PI)/4.0) < PI - ang) Angle.at<uchar>(i+1,j+1) = (uchar)4;
				else Angle.at<uchar>(i+1,j+1) = (uchar)0;
			}
			else if(ang >= -PI && ang < -((3.0*PI)/4.0)){
				if(ang + PI < -((3.0*PI)/4.0) - ang) Angle.at<uchar>(i+1,j+1) = (uchar)0;
				else Angle.at<uchar>(i+1,j+1) = (uchar)3;
			}
			else if(ang >= -((3.0*PI)/4.0) && ang < -PI/2.0){
				if(ang + ((3.0*PI)/4.0) < -(PI/2.0) - ang) Angle.at<uchar>(i+1,j+1) = (uchar)3;
				else Angle.at<uchar>(i+1,j+1) = (uchar)1;
			}
			else if(ang >= -PI/2.0 && ang < -PI/4.0){
				if(ang + (PI/2.0) < (-PI/4.0) - ang) Angle.at<uchar>(i+1,j+1) = (uchar)1;
				else Angle.at<uchar>(i+1,j+1) = (uchar)4;
			}
			else if(ang >= -PI/4.0 && ang < 0.0){
				if(ang + (PI/4.0) < -ang) Angle.at<uchar>(i+1,j+1) = (uchar)4;
				else Angle.at<uchar>(i+1,j+1) = (uchar)0;
			}
		}
	}
}

void canny::non_maximum_supression(){
	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			Mat window_gradient(G, Rect(j, i, 3, 3));
			Mat window_angle(Angle, Rect(j, i, 3, 3));

			//angle = 0 => x; angle = 1 => y; angle = 3 => xy; angle = 4 => -xy
			if(window_angle.at<uchar>(1,1) == (uchar)0){
				if(window_gradient.at<float>(1,1) > window_gradient.at<float>(2,1) &&
					window_gradient.at<float>(1,1) > window_gradient.at<float>(0,1)){
					G_supressed.at<float>(i+1,j+1) = window_gradient.at<float>(1,1);
					supression_mask.at<float>(i+1,j+1) = 255;
				}
			}
			else if(window_angle.at<uchar>(1,1) == (uchar)1){
				if(window_gradient.at<float>(1,1) > window_gradient.at<float>(1,2) &&
					window_gradient.at<float>(1,1) > window_gradient.at<float>(1,0)){
					G_supressed.at<float>(i+1,j+1) = window_gradient.at<float>(1,1);
					supression_mask.at<float>(i+1,j+1) = 255;
				}
			}
			else if(window_angle.at<uchar>(1,1) == (uchar)3){
				if(window_gradient.at<float>(1,1) > window_gradient.at<float>(0,2) &&
					window_gradient.at<float>(1,1) > window_gradient.at<float>(2,0)){
					G_supressed.at<float>(i+1,j+1) = window_gradient.at<float>(1,1);
					supression_mask.at<float>(i+1,j+1) = 255;
				}
			}
			else if(window_angle.at<uchar>(1,1) == (uchar)4){
				if(window_gradient.at<float>(1,1) > window_gradient.at<float>(0,0) &&
					window_gradient.at<float>(1,1) > window_gradient.at<float>(2,2)){
					G_supressed.at<float>(i+1,j+1) = window_gradient.at<float>(1,1);
					supression_mask.at<float>(i+1,j+1) = 255;
				}
			}
		}
	}
}

void canny::get_strong_edges(float high_threshold){
	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			Mat window_gradient(G_supressed, Rect(j, i, 3, 3));

			if(window_gradient.at<float>(1,1) != 0 && window_gradient.at<float>(1,1) > high_threshold){
				candidates.push_back({i+1,j+1});
				strong_edges.at<uchar>(i+1,j+1) = 255;
			}
		}
	}
}

void canny::get_weak_edges(float low_threshold, float high_threshold){
	for(int i=0;i<I_gray.rows - 3;i++){
		for(int j=0;j<I_gray.cols - 3;j++){
			Mat window_gradient(G_supressed, Rect(j, i, 3, 3));

			if(window_gradient.at<float>(1,1) != 0 && window_gradient.at<float>(1,1) <= high_threshold &&
			 window_gradient.at<float>(1,1) >= low_threshold){
			 	G_candidates.at<float>(i+1,j+1) = window_gradient.at<float>(1,1);
			 	weak_edges.at<uchar>(i+1,j+1) = 255;
			}
		}
	}
}


void canny::dfs(pair<int,int> p){
	visit.at<uchar>(p.first,p.second) = 1;

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			if(!(i == 0 && j == 0) && !(p.first + d[i] < 0 || p.first + d[i] > I_gray.rows - 1) &&
			 !(p.second + d[j] < 0 || p.second + d[j] > I_gray.cols - 1)){
				if(G_candidates.at<float>(p.first + d[i], p.second + d[j]) != 0 &&
				 visit.at<uchar>(p.first + d[i], p.second + d[j]) == 0) dfs({p.first + d[i], p.second + d[j]});
			}
		}
	}

	canny_vector.push_back(p);
}

void canny::blob_analysis(){
	for(int i=0;i<candidates.size();i++){
		if(visit.at<uchar>(candidates[i].first,candidates[i].second) == 0){
			dfs(candidates[i]);
		}
	}

	for(int i=0;i<canny_vector.size();i++) canny_result.at<uchar>(canny_vector[i].first,canny_vector[i].second) = 255; 
}
