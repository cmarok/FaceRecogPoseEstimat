#pragma once
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>

using namespace cv;
using namespace std;

Mat LBPHistograms(Mat LBP_face, int level);
vector<vector<vector<Mat>>> FacesLBP(vector<vector<vector<Mat>>> &Faces);
