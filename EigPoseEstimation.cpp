#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "EigPoseEstimation.h"
#include <iomanip>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;



// Pose Estimation Using Eigenfaces by Casimir


vector<vector<Mat>> getImagesByPoseClass(vector<vector<vector<Mat>>> &Faces);
vector<vector<Mat>> getHpSetByPoseClass(vector<Mat> &hp_dataset, vector<vector<int>> hp_labels);
Mat flatten(vector<Mat> poseClass);
void createPCAMatrix(vector<Mat> &data, Mat &dataMatrix);
void computeEigenfaces(Mat &training, Mat &eigenfaces, Mat &mean_im, int num);
void project(Mat &vectors, Mat &basis, Mat &projection);
double reconError(Mat &training, Mat &testing, int num);

EigPoseEstimation::EigPoseEstimation(vector<vector<vector<Mat>>> &Faces, vector<Mat> &hp_dataset, vector<vector<int>> hp_labels)
{

    vector<vector<Mat>> pose_classes = getImagesByPoseClass(Faces);
    vector<vector<Mat>> hp_classes = getHpSetByPoseClass(hp_dataset, hp_labels);

    vector<vector<Mat>> eigenSpace_representations;
    vector<Mat> eigenspaces;
    vector<Mat> means;


// get eigen info

    for (int i = 0; i < pose_classes.size(); ++i)
    {
        vector<Mat> testEig = pose_classes[i];
        Mat flattened;
        createPCAMatrix(testEig, flattened);
        Mat eigfaces, mean;
        computeEigenfaces(flattened, eigfaces, mean, 20);
        vector<Mat> class_projections;

        for (int j = 0; j < flattened.rows; ++j)
        {
            Mat training_image = flattened.row(j);
            Mat projection = (training_image - mean) * eigfaces.t();
            class_projections.push_back(projection);
        }
        eigenSpace_representations.push_back(class_projections);
        eigenspaces.push_back(eigfaces.t());
        means.push_back(mean);
    }



// buid matrix

    eig_confusion_matrix = Mat_<double>(21,21);
    for (int i = 0; i < 21; ++i)
    {
        for (int j = 0; j < 21; ++j)
        {
            eig_confusion_matrix.at<double>(i, j) = 0.0;
        }
    }


    for (int i = 0; i < 21; ++i)
    {
        for (int k = 0; k < hp_classes[i].size(); k++) {
            double min_distance = 10000;
            int class_index;
            Mat flatTestingImage = hp_classes[i][k].clone().reshape(0, 1);
            cvtColor(flatTestingImage, flatTestingImage, CV_RGB2GRAY);
            flatTestingImage.convertTo(flatTestingImage, CV_64F);

            for (int j = 0; j < 21; ++j)
            {
                // get eigenspaces and mean at index j
                Mat eigV = eigenspaces[j];
                Mat mean = means[j];
                // represent image[i][k] in the eigenspace
                Mat projected_testImage = (flatTestingImage - mean) * eigV;

                for (int l = 0; l < eigenSpace_representations[20-j].size(); ++l)
                {
                    double distance = norm(projected_testImage, eigenSpace_representations[20-j][l], NORM_L2);
                    // find the norm between test and train
                    if (distance < min_distance) {
                        min_distance = distance;
                        class_index = j;
                    }
                    // determine if norm is smallest or not and update class index to be j
                }
            }
            eig_confusion_matrix.at<double>(i, class_index) = eig_confusion_matrix.at<double>(i, class_index) + 1;
        }
    }


// print matrix

    vector<double> sums = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < 21; ++i)
    {
        for (int j = 0; j < 21; ++j)
        {
            sums[i] = sums[i] + eig_confusion_matrix.at<double>(i, j);
        }
    }

    for (int i = 0; i < 21; ++i)
    {
        for (int j = 0; j < 21; ++j)
        {
            eig_confusion_matrix.at<double>(i, j) = eig_confusion_matrix.at<double>(i, j)/sums[i];
        }
    }

    for (int i = 0; i < 21; ++i)
    {
        for (int j = 0; j < 21; ++j)
        {
            if (j == 20)
            {
                printf("%.2f\n", eig_confusion_matrix.at<double>(i,j));
            }
            else {
                printf("%.2f,", eig_confusion_matrix.at<double>(i,j));
            }
        }
    }
    




}

vector<vector<Mat>> getImagesByPoseClass(vector<vector<vector<Mat>>> &Faces) 
{   
    vector<vector<Mat>> QMUL_poses;
    vector<vector<int>> tilts = {{0,1},{2,3,4},{5,6}};
    vector<vector<int>> pans = {{0,1},{2,3,4},{5,6,7},{8,9,10},{11,12,13},{14,15,16},{17,18}};


        int index = 0;
        for (int j = 0; j < tilts.size(); ++j)
        {
            for (int k = 0; k < pans.size(); ++k)
            {
                vector<Mat> class_vec;
                for (int l = 0; l < tilts[j].size(); ++l)
                {
                    for (int m = 0; m < pans[k].size(); ++m)
                    {
                        for (int i = 0; i < 29; ++i)
                        {
                            Mat subclass;
                            subclass = Faces[i][tilts[j][l]][pans[k][m]].clone();
                            if (subclass.empty()) {
                                cout << tilts[j][l] << endl;
                                cout << pans[k][m] << endl;
                                cout << i << endl;

                            }
                            class_vec.push_back(subclass);
                        }
                    }
                    
                }
                QMUL_poses.push_back(class_vec);
            }
        }
    // }
    return QMUL_poses;

}



vector<vector<Mat>> getHpSetByPoseClass(vector<Mat> &hp_dataset, vector<vector<int>> hp_labels) 
{   
    vector<vector<Mat>> hp_poses;
    vector<Mat> class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11,class12,class13,class14,class15,class16,class17,class18,class19,class20,class21;


        int index = 0;
        for (int i = 0; i < hp_dataset.size(); ++i)
        {
            if(hp_labels[i][0] == 0 || hp_labels[i][0] == 1) {
                if(hp_labels[i][1] == 0) {
                    class1.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 1 || hp_labels[i][1] == 2) {
                    class2.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 3 || hp_labels[i][1] == 4) {
                    class3.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 5 || hp_labels[i][1] == 6 || hp_labels[i][1] == 7) {
                    class4.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 8 || hp_labels[i][1] == 9) {
                    class5.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 10 || hp_labels[i][1] == 11) {
                    class6.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 12 || hp_labels[i][1] == 13) {
                    class7.push_back(hp_dataset[i]);
                }
            }
            else if(hp_labels[i][0] == 2) {
                if(hp_labels[i][1] == 0) {
                    class8.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 1 || hp_labels[i][1] == 2) {
                    class9.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 3 || hp_labels[i][1] == 4) {
                    class10.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 5 || hp_labels[i][1] == 6 || hp_labels[i][1] == 7) {
                    class11.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 8 || hp_labels[i][1] == 9) {
                    class12.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 10 || hp_labels[i][1] == 11) {
                    class13.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 12 || hp_labels[i][1] == 13) {
                    class14.push_back(hp_dataset[i]);
                }
            }
            else if(hp_labels[i][0] == 3 || hp_labels[i][0] == 4) {
                if(hp_labels[i][1] == 0) {
                    class15.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 1 || hp_labels[i][1] == 2) {
                    class16.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 3 || hp_labels[i][1] == 4) {
                    class17.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 5 || hp_labels[i][1] == 6 || hp_labels[i][1] == 7) {
                    class18.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 8 || hp_labels[i][1] == 9) {
                    class19.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 10 || hp_labels[i][1] == 11) {
                    class20.push_back(hp_dataset[i]);
                }
                else if (hp_labels[i][1] == 12 || hp_labels[i][1] == 13) {
                    class21.push_back(hp_dataset[i]);
                }
            }
        }

        hp_poses.push_back(class21);
        hp_poses.push_back(class20);
        hp_poses.push_back(class19);
        hp_poses.push_back(class18);
        hp_poses.push_back(class17);
        hp_poses.push_back(class16);
        hp_poses.push_back(class15);
        hp_poses.push_back(class14);
        hp_poses.push_back(class13);
        hp_poses.push_back(class12);
        hp_poses.push_back(class11);
        hp_poses.push_back(class10);
        hp_poses.push_back(class9);
        hp_poses.push_back(class8);
        hp_poses.push_back(class7);
        hp_poses.push_back(class6);
        hp_poses.push_back(class5);
        hp_poses.push_back(class4);
        hp_poses.push_back(class3);
        hp_poses.push_back(class2);
        hp_poses.push_back(class1);

    return hp_poses;

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


// Compute PCA eigenfaces
void computeEigenfaces(Mat &training, Mat &eigvectors, Mat &mean_im, int num){

    // PCA object
    PCA pca(training, noArray(), CV_PCA_DATA_AS_ROW, num);

    // Compute and display mean (Question 3)
    mean_im = pca.mean;
    eigvectors = pca.eigenvectors;
}





