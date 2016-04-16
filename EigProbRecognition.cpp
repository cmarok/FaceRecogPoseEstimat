#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "EigProbRecognition.h"
#include <iomanip>

using namespace std;
using namespace cv;



// Probabilistic Face recognition Using Eigenfaces by Casimir


vector<Mat> createBigFatVector(vector<vector<vector<Mat>>> &Faces);
void createPCAMatrixProb(vector<Mat> &data, Mat &dataMatrix);
void computeEigenfacesProb(Mat &training, Mat &eigenfaces, Mat &mean_im, int num);
void partitionImagesProb(vector<Mat> &faceSet, vector<Mat> &training, vector<Mat> &testing, int k, int fold);
vector<double> getLikelihoods(Mat &data, Mat &mean, Mat &covar, double det);

EigProbRecognition::EigProbRecognition(vector<vector<vector<Mat>>> &Faces)
{

    vector<Mat> image_set = createBigFatVector(Faces);

    vector<Mat> testImageSet;
    for (int i = 0; i < 20; ++i)
    {
        testImageSet.push_back(image_set[i]);
    }



    random_shuffle(testImageSet.begin(), testImageSet.end());
    Mat training, testing;
    vector<Mat> train, test;
    partitionImagesProb(testImageSet, train, test, 7, rand()%7);
    createPCAMatrixProb(train, training);
    createPCAMatrixProb(test, testing);

    // get eigenspaces, mean, covariance matrix and det of covariance matrix
    Mat eigenspace;
    Mat mean;
    Mat covar, mean_2;
    computeEigenfacesProb(training, eigenspace, mean, 20);
    calcCovarMatrix(training.t(), covar, mean_2, CV_COVAR_SCRAMBLED| CV_COVAR_ROWS);
    // vector<double> likelihoods = getLikelihoods(training, mean, covar, det);

    



}

vector<Mat> createBigFatVector(vector<vector<vector<Mat>>> &Faces)
{
    vector<Mat> bigFatVector;
    for (int i = 0; i < Faces.size(); ++i)
    {
        for (int j = 0; j < Faces[i].size(); ++j)
        {
            for (int k = 0; k < Faces[i][j].size(); ++k)
            {
                bigFatVector.push_back(Faces[i][j][k]);
            }
        }
    }
    return bigFatVector;
}


// Create a data matrix, each row containing a vectorized training image
void createPCAMatrixProb(vector<Mat> &data, Mat &dataMatrix){
    for (int i = 0; i < data.size(); i++){
        Mat imgVector = data[i].reshape(0, 1);
        dataMatrix.push_back(imgVector);
    }
    cvtColor(dataMatrix, dataMatrix, CV_RGB2GRAY);
    dataMatrix.convertTo(dataMatrix, CV_64F);
}


// Compute PCA eigenfaces
void computeEigenfacesProb(Mat &training, Mat &eigvectors, Mat &mean_im, int num){

    // PCA object
    PCA pca(training, noArray(), CV_PCA_DATA_AS_ROW, num);

    // Compute and display mean (Question 3)
    mean_im = pca.mean;
    eigvectors = pca.eigenvectors;
}

// Partition the images into training and testing images for k-fold cross validation
void partitionImagesProb(vector<Mat> &faceSet, vector<Mat> &training, vector<Mat> &testing, int k, int fold){
    for (int i = 0; i < faceSet.size(); i++){
        if ((i%k) == fold){
            testing.push_back(faceSet[i]);
        }
        else{
            training.push_back(faceSet[i]);
        }
    }
}


vector<double> getLikelihoods(Mat &data, Mat &mean, Mat &covar, double det)
{
    vector<double> likelihoods;

    for (int i = 0; i < data.rows; ++i)
    {
        Mat trImage = data.row(i);
        Mat normd = trImage - mean;
        Mat exp_index = normd*covar*normd.t(); 
        double final_exp = exp(exp_index.at<double>(0, 0)/(-2));
        double likelihood = (1/pow((3.141592653*det*2), 2.0/(double)trImage.cols))*final_exp;
        likelihoods.push_back(likelihood);
    }
    return likelihoods;
}














