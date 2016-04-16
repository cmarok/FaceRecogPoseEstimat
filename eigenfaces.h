#ifndef eigenfaces_h
#define eigenfaces_h

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "bow.h"

using namespace cv;
using namespace std;

void createPoseSet(vector<vector<vector<Mat>>> &Faces, vector<vector<Mat>> &PoseSet);
void createFaceSet(vector<vector<vector<Mat>>> &Faces, vector<Mat> &faceSet);
void partitionImages(vector<Mat> &faceSet, vector<Mat> &training, vector<Mat> &testing, int k, int fold);
void createPCAMatrix(vector<Mat> &data, Mat &dataMatrix);
double computeEigenfaces(Mat &training, Mat &testing, Mat &eigenfaces, int numVectors);
Mat project(Mat &data, Mat &base, Mat &mean);
Mat backProject(Mat &data, Mat &base, Mat &mean);

#endif