#ifndef EigPoseEstimation_H
#define EigPoseEstimation_H

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

class EigPoseEstimation
{
private:
    Mat eig_confusion_matrix;


public:


    EigPoseEstimation(vector<vector<vector<Mat>>> &Faces, vector<Mat> &hp_dataset, vector<vector<int>> hp_labels);


};

#endif // EigPoseEstimation_H
