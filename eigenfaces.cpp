#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "eigenfaces.h"

#define IMSHOW 0;

using namespace cv;
using namespace std;

// Rearrange QMUL images, sorting by pose class
// Images in the output vector can be accessed as: poseSet[poseClass][subjectNumber]
void createPoseSet(vector<vector<vector<Mat>>> &Faces, vector<vector<Mat>> &poseSet){

	for (int i = 0; i < Faces[0].size(); i++){
		for (int j = 0; j < Faces[0][0].size(); j++){
			vector<Mat> currentPose;
			for (int k = 0; k < Faces.size(); k++){
				currentPose.push_back(Faces[k][i][j]);
			}
			poseSet.push_back(currentPose);
		}
	}
}

// Rearrange QMUL images, putting every image in a single vector
void createFaceSet(vector<vector<vector<Mat>>> &Faces, vector<Mat> &faceSet){

	for (int i = 0; i < Faces.size(); i++){
		for (int j = 0; j < Faces[i].size(); j++){
			for (int k = 0; k < Faces[i][j].size(); k++){
				faceSet.push_back(Faces[i][j][k]);
			}
		}
	}
}

// Partition the images into training and testing images for k-fold cross validation
void partitionImages(vector<Mat> &faceSet, vector<Mat> &training, vector<Mat> &testing, int k, int fold){
	for (int i = 0; i < faceSet.size(); i++){
		if ((i%k) == fold){
			testing.push_back(faceSet[i]);
		}
		else{
			training.push_back(faceSet[i]);
		}
	}
}

// Create a data matrix, each row containing a vectorized training image
void createPCAMatrix(vector<Mat> &data, Mat &dataMatrix){
	for (int i = 0; i < data.size(); i++){
		Mat imgVector = data[i].reshape(0, 1);
		dataMatrix.push_back(imgVector);
	}
	cvtColor(dataMatrix, dataMatrix, CV_RGB2GRAY);
	dataMatrix.convertTo(dataMatrix, CV_64F);
}

// Projects an image onto a base
Mat project(Mat &data, Mat &base, Mat &mean){
	Mat projection;
	Mat tmp_data, tmp_mean = repeat(mean, data.rows / mean.rows, data.cols / mean.cols);
	data.convertTo(tmp_data, mean.type());
	subtract(tmp_data, tmp_mean, tmp_data);
	gemm(tmp_data, base, 1, Mat(), 0, projection, GEMM_2_T);

	return projection;
}

// Projects back from projected image
Mat backProject(Mat &data, Mat &base, Mat &mean){
	Mat backProjection;
	Mat tmp_data, tmp_mean;
	data.convertTo(tmp_data, mean.type());
	tmp_mean = repeat(mean, data.rows, 1);
	gemm(tmp_data, base, 1, tmp_mean, 1, backProjection, 0);

	return backProjection;
}


// Compute PCA eigenfaces and returns reconstruction error
double computeEigenfaces(Mat &training, Mat &testing, Mat &eigenfaces, int numComponents){

	// Compute mean and subtract from each row
	Mat mean = Mat(1, training.cols, training.type());
	reduce(training, mean, 0, CV_REDUCE_AVG);
	for (int i = 0; i<training.rows; i++) {
		training.row(i) -= mean;
	}
	Mat newMean;
	mean.convertTo(newMean, CV_8U);

	// Display mean (Question 3)
#if IMSHOW
	imshow("mean", newMean.reshape(1, 100));
	waitKey(0);
#endif

	// Compute covariance matrix and eigenfunction
	Mat covarMean, covar, eigenvalues, covarEigenvectors;
	calcCovarMatrix(training, covar, covarMean, CV_COVAR_SCALE | CV_COVAR_SCRAMBLED | CV_COVAR_ROWS);
	eigen(covar, eigenvalues, covarEigenvectors);
	Mat eigenvectors = covarEigenvectors.t()*training;

	// Display first 10 eigenfaces  (Question 3)
	for (int i = 0; i < numComponents; i++){
		Mat eigenface = eigenvectors.row(i);
		eigenfaces.push_back(eigenface);
		normalize(eigenface, eigenface, 0, 255, NORM_MINMAX, CV_8UC1);

#if IMSHOW
		namedWindow("eigenface", CV_WINDOW_AUTOSIZE);
		moveWindow("eigenface", 100, 100);
		imshow("eigenface", eigenface.reshape(1, 100));
		waitKey(0);
		destroyWindow("eigenface");
#endif

	}

	// Reconstruction error. 2 random images are chosen for display for original and reconstructed

	double error = 0;
	for (int i = 0; i < testing.rows; i++){

		// projecting and backprojecting

		Mat projection = project(testing.row(i), eigenfaces, mean);
		Mat reconstructed = backProject(projection, eigenfaces, mean);
		double reconError = norm(testing.row(i), reconstructed, NORM_L2);

		/*		normalize(reconstructed, reconstructed, 0, 255, NORM_MINMAX, CV_8UC1);
		namedWindow("reconstructed", CV_WINDOW_AUTOSIZE);
		moveWindow("reconstructed", 100, 100);
		imshow("reconstructed", reconstructed.reshape(1, 100));
		waitKey(0);
		destroyWindow("reconstructed"); */


		error += reconError;
	}

	error = error / testing.rows;
	return error;


}

// Compute k-fold cross validation average reconstruction error
double avgReconError(vector<Mat> faces, int numComponents, int k, bool isTest){

	double totalError = 0.0;
	Mat eigenfaces;
	for (int i = 0; i < k; i++){
		vector<Mat> train, test;
		Mat training, testing;
		partitionImages(faces, train, test, k, i);
		createPCAMatrix(train, training);
		createPCAMatrix(test, testing);

		double error = 0.0;

		if (isTest){
			error = computeEigenfaces(training, testing, eigenfaces, numComponents);
		}
		else{
			error = computeEigenfaces(training, training, eigenfaces, numComponents);
		}
		totalError += error;
	}
	double avgError = totalError / k;
	return avgError;
}


// Compute recognition rate
double recogRate(Mat &training, Mat &testing, vector<double> &labels, int numComponents){
	Mat trainingEigenFaces, testingEigenFaces;
}