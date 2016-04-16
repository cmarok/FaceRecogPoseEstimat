

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
