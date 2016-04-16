

//
//  main.cpp
//  Project
//
//  Created on 19/03/2016.
//  Copyright © 2016 project. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "bow.h"
#include "headpose.h"
#include "EigPoseEstimation.h"
#include "EigProbRecognition.h"
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

void createPoseSet(vector<vector<vector<Mat>>> &Faces, vector<vector<Mat>> &PoseSet);
void poseEstimationSet(vector<vector<vector<Mat>>> &Faces, vector<vector<vector<Mat>>> &poseSet);
void createFaceSet(vector<vector<vector<Mat>>> &Faces, vector<Mat> &faceSet);
void partitionImages(vector<Mat> &faceSet, vector<Mat> &training, vector<Mat> &testing, int k, int fold);

void LoadQMUL(vector<vector<vector<Mat>>> &Faces);
void LoadQMUL_cas(vector<vector<vector<Mat>>> &Faces_cas);
void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number);
double computePixelLBP(const Mat input);
vector<vector<vector<Mat>>> PosesLBP(vector<vector<vector<vector<Mat>>>> &Pose);
Mat LBP_pose_est(vector<vector<vector<Mat>>> Histograms, HeadPose hp, int level);

int main(int argc, const char * argv[]) 
{    
    //faces array
    vector<vector<vector<Mat>>> Faces;
    vector<vector<vector<Mat>>> Faces2;

    vector<vector<vector<Mat>>> Pose;
    LoadQMUL(Faces);

    // I get discontinous data sor some reason, so I just adapted the method for me
    LoadQMUL_cas(Faces2);
    
    //choose who to display
    int subject_number = 30;//index
    
    //display subject function
    // Display_subject(Faces,subject_number);

    const string cas_path = "/Users/casimirdesarmeaux/Documents/McGill/ECSE-415/Project/HeadPoseImageDatabase/";
    const string meong_path = "/Users/MeongheeSeo/Documents/2016 Winter/ECSE 415/Projects/HeadPoseImageDatabase/";

    HeadPose hp = HeadPose(cas_path);
    // hp.displayImages(2, 15);

    // ---- EIGENFACE POSE ESTIMATION ---UNCOMMENT TO ENABLE
    //EigPoseEstimation epe = EigPoseEstimation(Faces2, hp.hp_dataset, hp.hp_labels);

    // ---- EIGENFACE PROBABILISTIC FACE RECOGNITION ---UNCOMMENT TO ENABLE
    //EigProbRecognition eigProb = EigProbRecognition(Faces2);

    // --------- BOW TESTING ----------------------
//    initModule_nonfree();
//    int codeWords = 100;
//    BOW::recognition(Faces, codeWords);
//    BOW::recognitionP(Faces, codeWords);
//    vector<vector<vector<Mat>>> pose;
//    poseEstimationSet(Faces, pose);
//    BOW::poseRecognition(pose, hp);
    // --------------------------------------------
    return 0;
}


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

// Rearrange QMUL images for pose estimation
void poseEstimationSet(vector<vector<vector<Mat>>> &Faces, vector<vector<vector<Mat>>> &poseSet)
{
    int subject_size = Faces.size();
    int tilt_size = Faces[0].size();
    int pan_size = Faces[0][0].size();

    for (int t = 0; t < tilt_size; t++) {
        vector<vector<Mat>> pan_vec;

        for (int p = 0; p < pan_size; p+=3) {
            vector<Mat> sub_vec;

            for (int s = 0; s <subject_size; s++) {
                if (p == 0) {
                    sub_vec.push_back(Faces[s][t][p]);
                    sub_vec.push_back(Faces[s][t][p+1]);
                }
                else if (p == pan_size-1) {
                    sub_vec.push_back(Faces[s][t][p-1]);
                    sub_vec.push_back(Faces[s][t][p]);
                }
                else {
                    sub_vec.push_back(Faces[s][t][p-1]);
                    sub_vec.push_back(Faces[s][t][p]);
                    sub_vec.push_back(Faces[s][t][p+1]);
                }
            }
            pan_vec.push_back(sub_vec);
        }

        if (t == 1 || t == 4 || t == 6) {
            poseSet.push_back(pan_vec);
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



void LoadQMUL(vector<vector<vector<Mat>>> &Faces)
{
    
    //change to location of grey dataset when running on another machine
    const string cas_path = "/Users/casimirdesarmeaux/Documents/McGill/ECSE-415/Project/QMUL/";
    const string meong_path = "/Users/MeongheeSeo/Documents/2016 Winter/ECSE 415/Projects/QMUL/Set1_Greyscale/";    
    const string Dataset_location = cas_path;
    
    const vector<string> Subject_names = {"AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ", "DennisP",
                                          "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack",
                                          "JamieS", "JeffN", "John", "Jon", "KateS", "KatherineW", "KeithC",
                                          "KrystynaN", "PaulV", "RichardB", "RichardH",
                                          "SarahL", "SeanG", "SeanGNoGlasses", "SimonB", "SueW", "TasosH", "TomK",
                                          "YogeshR", "YongminY"};
    const vector<string> Subject_names2 = {"AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ", "DennisP",
                                           "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack",
                                           "JamieS", "JeffN", "John", "Jon", "KateS", "KatherineW", "KeithC",
                                           "KrystynaN", "PaulV", "RichardB", "RichardH", "SarahL",
                                           "SeanG", "SeanGNoGlasses", "SimonB", "SueW", "TasosH", "TomK", "YogeshR",
                                           "YongminY"};
    const vector<string> Tilt_code = {"060", "070", "080", "090", "100", "110", "120"};
    const vector<string> Pan_code = {"000", "010", "020", "030", "040", "050", "060", "070", "080", "090",
                                     "100", "110", "120", "130", "140", "150", "160", "170", "180"};
    //there may be nicer ways of doing it but this was the fastest to code (and i expect run since no int->str conversions etc)
    
    //loops load all the images
    for (int i = 0; i < Subject_names.size(); i++){
        vector<vector<Mat>> Tilting;
        for (int j = 0; j < Tilt_code.size(); j++){
            vector<Mat> Panning;
            for (int k=0; k<Pan_code.size();k++){
                string temp = Dataset_location + Subject_names[i] + "Grey/" + Subject_names2[i] + "_" +
                        Tilt_code[j] + "_" + Pan_code[k] + ".ras";
                
                Panning.push_back(imread(temp));
            }

            Tilting.push_back(Panning);
        }
        
        Faces.push_back(Tilting);
    }
}


void LoadQMUL_cas(vector<vector<vector<Mat>>> &Faces_cas)
{
    
    //change to location of grey dataset when running on another machine
    const string cas_path = "/Users/casimirdesarmeaux/Documents/McGill/ECSE-415/Project/QMUL/";
    const string meong_path = "/Users/MeongheeSeo/Documents/2016 Winter/ECSE 415/Projects/QMUL/Set1_Greyscale/";    
    const string Dataset_location = cas_path;
    
    const vector<string> Subject_names = {"AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ", "DennisP",
                                          "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack",
                                          "JamieS", "John", "KateS", "KatherineW", "KeithC",
                                          "KrystynaN", "PaulV", "RichardB", "RichardH",
                                          "SarahL", "SeanG", "SeanGNoGlasses", "SimonB", "SueW", "TasosH", "TomK",
                                          "YogeshR", "YongminY"};
    const vector<string> Subject_names2 = {"AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ", "DennisP",
                                           "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack",
                                           "JamieS", "John","KateS", "KatherineW", "KeithC",
                                           "KrystynaN", "PaulV", "RichardB", "RichardH", "SarahL",
                                           "SeanG", "SeanGNoGlasses", "SimonB", "SueW", "TasosH", "TomK", "YogeshR",
                                           "YongminY"};
    const vector<string> Tilt_code = {"060", "070", "080", "090", "100", "110", "120"};
    const vector<string> Pan_code = {"000", "010", "020", "030", "040", "050", "060", "070", "080", "090",
                                     "100", "110", "120", "130", "140", "150", "160", "170", "180"};
    //there may be nicer ways of doing it but this was the fastest to code (and i expect run since no int->str conversions etc)
    
    //loops load all the images
    for (int i = 0; i < Subject_names.size(); i++){
        vector<vector<Mat>> Tilting;
        for (int j = 0; j < Tilt_code.size(); j++){
            vector<Mat> Panning;
            for (int k=0; k<Pan_code.size();k++){
                string temp = Dataset_location + Subject_names[i] + "Grey/" + Subject_names2[i] + "_" +
                        Tilt_code[j] + "_" + Pan_code[k] + ".ras";
                
                Panning.push_back(imread(temp));
            }

            Tilting.push_back(Panning);
        }
        
        Faces_cas.push_back(Tilting);
    }
}



void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number)
{
    Mat Display_img;
    for (int j = 0; j < 7; j++){
        Mat Row = Faces[Subject_number][j][0];
        
        for (int k = 1; k < 19; k++){
            Mat red = Faces[Subject_number][j][k];
            Mat Row2;
            hconcat(Row, red, Row2);
            Row = Row2; 
        }
        
        if (j == 0) {
            Display_img = Row;
        } else {
            vconcat(Display_img, Row,Display_img);
        }
    }
    
    imshow("hello", Display_img);
    namedWindow( "hello", WINDOW_AUTOSIZE );
    waitKey(0);
}

//converts faces to LBP images
vector<vector<vector<Mat>>> FacesLBP(vector<vector<vector<Mat>>> &Faces){
    vector<vector<vector<Mat>>> LBP_faces = Faces;

    
    //for all of the images in the object
    for (int k=0; k<Faces.size();k++){
        for(int l=0;l<Faces[k].size();l++){
            for(int m=0; m<Faces[k][l].size();m++){
                
                Mat image1 =Faces[k][l][m];
                Mat image_out(image1.size(), CV_64F);
    
                
                //for all of the pixels (not including border pixels)
                for (int i = 1; i < image1.cols-1; i++){
                    for (int j = 1; j < image1.rows-1; j++){
                        //extract surrounding 3x3 matrix for every pixel
                        Mat pixelNeighbors = image1(Rect(i - 1, j - 1, 3, 3));
                        
                        //comupte lbp
                        image_out.at<double>(j, i) = computePixelLBP(pixelNeighbors);
                    }
                }
                image_out.convertTo(image_out, CV_8U);
                LBP_faces[k][l][m] = image_out;
            }
        }
    }

    return LBP_faces;
}

double computePixelLBP(const Mat input){
    //compute LBP value for pixel
    Mat pixel;
    
    input.convertTo(pixel, CV_32S);

    
    int center_cell = pixel.at<int>(1, 1);
    vector<int> Binary_vec;
    
    //check every pixel and create 8bit array, starting at left top corner of matrix
    Binary_vec.push_back(!(center_cell < pixel.at<int>(0, 0)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(0, 1)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(0, 2)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(1, 2)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(2, 2)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(2, 1)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(2, 0)));
    Binary_vec.push_back(!(center_cell < pixel.at<int>(1, 0)));
    
    //make sure less than 2 transitions
    int transitions = 0;
    for (int i = 0; i < Binary_vec.size()-1; i++){
        if (Binary_vec[i + 1] - Binary_vec[i] != 0)
            transitions = transitions + 1;
    }
    if (Binary_vec[0] - Binary_vec[Binary_vec.size()-1] != 0){
        transitions = transitions + 1;
    }
    //get LBP value
    double LVPvalue = 0;
    //if transitions are 2 or less, compute the LBP value, otherwise LVPvalue remains 0
    if (transitions <= 2){
        for (int i = 0; i < Binary_vec.size(); i++){
            if (Binary_vec[i] == 1){
                LVPvalue = LVPvalue + pow(2, (double)i);
            }
        }
    }
    //cout << LVPvalue << "\n";
    //return LVP value
    return LVPvalue;
    
}


//divides LBP images into boxes and finds histograms for the boxes, concatantes them for each image
//level=0 -> whole image (1x1)
//level=1 -> 4 boxes  (2x2)
//level=2 -> 16 boxes (4x4) etc
vector<vector<vector<Mat>>> LBPHistograms(vector<vector<vector<Mat>>> &LBP_face, int level){

    vector<vector<vector<Mat>>> ALL_Histograms = LBP_face;
    
    //goes through all the images in the datase/////////////////
    for (int k=0; k<LBP_face.size();k++){
        for(int l=0;l<LBP_face[k].size();l++){
            for(int m=0; m<LBP_face[k][l].size();m++){
                int divisions = pow(2,level);//number boxes per axis
                int width = LBP_face[k][l][m].cols;
                width = width/divisions; //width of each box
                int height = LBP_face[k][l][m].rows;
                height = height/divisions; //height of each box
                Mat Image_Histogram;
                
                //goes through each of the boxes
                for(int i=0; i<divisions; i++){
                    for(int j=0;j<divisions; j++){
                        
                        //get box
                        Mat patch = LBP_face[k][l][m];
                        patch = patch(Rect(i*width, j*height, width-1, height-1));
                        
                        //Histogram settings
                        Mat hist;
                        int bins = 59;
                        int histSize[] = {bins};
                        float lranges[] = {0, 256};
                        const float* ranges[] = {lranges};
                        
                        //get histogram of current box
                        calcHist(&patch, 1, 0, cv::Mat(), hist, 1, histSize, ranges);
                        
                        //concatanate it to the rest of the boxes in the image
                        if (i==0 && j==0){
                            Image_Histogram=hist;
                        }else{
                            hconcat(Image_Histogram,hist,Image_Histogram);
                        }
                    }
                }
                ALL_Histograms[k][l][m]= Image_Histogram;
            }
        }
    }
    return ALL_Histograms;
}


Mat LBP_pose_est(vector<vector<vector<Mat>>> Faces_hist, HeadPose hp,int level){
    Mat confusion = Mat::zeros(21,21,CV_64F);
    vector<vector<vector<vector<Mat>>>> Pose_hist;
    vector<string> tilt = {"-30", "-15", "+0", "+15", "+30"};
    vector<string> pan = {"-90", "-75", "-60", "-45", "-30", "-15", "+0", "+15", "+30", "+45", "+60", "+75", "+90"};
    
    vector<string> Pan_code = {"000","010","020","030","040","050","060","070","080","090","100","110","120","130","140", "150","160","170","180"};
    //only testing for 21 images
    for(int i=0;i<hp.images.size();i++){
        for(int j=0;j<hp.images[i].size();j++){
            for(int k=0;k<hp.images[i][j].size();k++){
                for (int l=0; l<hp.images[i][j][k].size(); l++) {
                    
                    if (l%2==0){//only if one of the 7 pan angles we are testing
                        int divisions = pow(2,level);//number boxes per axis
                        int width = hp.images[i][j][k][l].cols;
                        width = width/divisions; //width of each box
                        int height = hp.images[i][j][k][l].rows;
                        height = height/divisions; //height of each box
                        Mat Image_Histogram;
                        
                        //goes through each of the boxes
                        for(int p=0; p<divisions; p++){
                            for(int q=0;q<divisions; q++){
                                
                                //get box
                                Mat patch = hp.images[i][j][k][l];
                                patch = patch(Rect(p*width, q*height, width-1, height-1));
                                
                                //Histogram settings
                                Mat hist;
                                int bins = 59;
                                int histSize[] = {bins};
                                float lranges[] = {0, 256};
                                const float* ranges[] = {lranges};
                                
                                //get histogram of current box
                                calcHist(&patch, 1, 0, cv::Mat(), hist, 1, histSize, ranges);
                                
                                //concatanate it to the rest of the boxes in the image
                                if (p==0 && q==0){
                                    Image_Histogram=hist;
                                }else{
                                    hconcat(Image_Histogram,hist,Image_Histogram);
                                }
                            }
                        }
                        double dist, shortest_dist;
                        int best_match_tilt,best_match_pan;
                        
                        for(int m=0;m<5;m++){ //tilt angles
                            for (int n=0; n<19; n=n+3) {
                                for(int o=0;o<Faces_hist.size();o++){
                                    dist = compareHist(Faces_hist[o][m][n], Image_Histogram, CV_COMP_CHISQR);
                                    if(m==0 && n==0 && o==0){
                                        shortest_dist=dist;
                                        best_match_tilt = m;
                                        best_match_pan = n;
                                        
                                    }else if(shortest_dist>dist) {
                                        shortest_dist=dist;
                                        best_match_tilt = m;
                                        best_match_pan = n;
                                    
                                    
                                    }
                                }
                            }
                        
                        }
                    
                        if(best_match_tilt<2){
                            confusion.at<double>(l/2, best_match_pan)++;
                        }else if (best_match_tilt==2){
                            confusion.at<double>(l/2 + 7, best_match_pan+7)++;
                        }else{
                            confusion.at<double>(l/2 + 14, best_match_pan+14)++;
                        }
                        
                    }
                }
            }
        }
    }
    cout<<confusion<<endl;
    return confusion;
}

