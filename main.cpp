//
//  main.cpp
//  Project
//
//  Created on 19/03/2016.
//  Copyright © 2016 project. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;




void LoadQMUL(vector<vector<vector<Mat>>> &Faces);

void LoadHeadPose(vector<vector<vector<Mat>>> &Pose);

void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number);

int main(int argc, const char * argv[]) {
    
    
    
    //faces array
    vector<vector<vector<Mat>>> Faces;
    vector<vector<vector<Mat>>> Pose;
    LoadQMUL(Faces);
    LoadHeadPose(Pose);
    
    //choose who to display
    int subject_number = 35;//index
    
    //display subject function
    Display_subject(Faces,subject_number);
    return 0;
}


void LoadQMUL(vector<vector<vector<Mat>>> &Faces)
{
    
    //change to location of grey dataset when running on another machine
    const string Dataset_location = "/Users/user1/Desktop/uni_archive/semester5/ECSE415vision/project/QMUL_FaceDatabase/Set1_Greyscale/";
    
    const vector<string> Subject_names = { "AdamB",  "AdamBTest", "AndreeaV" ,"CarlaB","ColinP","DanJ","DennisP","DennisPNoGlasses","DerekC","DerekCTest","GrahamW","HeatherL","Jack", "JamieS","JamieSTest","JeffN","John","Jon","KateS","KatherineW","KeithC","KrystynaN","PaulV","RichardB", "RichardBTest","RichardH","RichardHTest","SarahL","SeanG","SeanGNoGlasses","SimonB","SueW","TasosH","TomK","YogeshR", "YongminY"};
    const vector<string> Subject_names2 = { "AdamB",  "AdamB", "AndreeaV" ,"CarlaB","ColinP","DanJ","DennisP","DennisPNoGlasses","DerekC","DerekC","GrahamW","HeatherL","Jack", "JamieS","JamieS","JeffN","John","Jon","KateS","KatherineW","KeithC","KrystynaN","PaulV","RichardB", "RichardB","RichardH","RichardH","SarahL","SeanG","SeanGNoGlasses","SimonB","SueW","TasosH","TomK","YogeshR", "YongminY"};
    const vector<string> Tilt_code = {"060","070","080","090","100","110","120"};
    const vector<string> Pan_code = {"000","010","020","030","040","050","060","070","080","090","100","110","120","130","140", "150","160","170","180"};
    //there may be nicer ways of doing it but this was the fastest to code (and i expect run since no int->str conversions etc)
    
    //loops load all the images
    for (int i = 0; i < Subject_names.size(); i++){
        vector<vector<Mat>> Tilting;
        for (int j=0; j<Tilt_code.size();j++){
            vector<Mat> Panning;
            for (int k=0; k<Pan_code.size();k++){
                string temp = Dataset_location + Subject_names[i]+"Grey/"+Subject_names2[i]+"_"+Tilt_code[j]+"_"+Pan_code[k]+".ras";
                
                //////////////////////////////////////////////////////////////////////////
                Panning.push_back(imread(temp));
                ///////////////////////////////////////////////////////////////////////////
                
                
            }
            Tilting.push_back(Panning);
        }
        
        Faces.push_back(Tilting);
    }
}




void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number)
{
    Mat Display_img;
    for (int j=0; j<7;j++){
        Mat Row = Faces[Subject_number][j][0];
        
        for (int k=1; k<19;k++){
            Mat red = Faces[Subject_number][j][k];
            Mat Row2;
            hconcat(Row,red,Row2);
            Row = Row2;
            
        }
        
        if (j==0){
            Display_img = Row;
        }else{
            vconcat(Display_img, Row,Display_img);
        }
        
        }
    
    imshow("hello", Display_img);
    namedWindow( "hello", WINDOW_AUTOSIZE );
//    imwrite("/Users/user1/Desktop/uni_archive/semester5/ECSE415vision/project/YongminY.jpg",Display_img);
    waitKey(0);
   
}


void LoadHeadPose(vector<vector<vector<Mat>>> &Pose){
    const string Dataset_location = "/Users/user1/Desktop/uni_archive/semester5/ECSE415vision/project/HeadPoseImageDatabase/";
    const vector<string> angle_code = {"-90","-75","-60","-45","-30","-15","+0","+15","+30","+45","+60","+75","+90"};
    int m=0;
    for (int i=0;i<15;i++){ //each person
        for (int j = 0; j<2;j++){ //each series
            for(int k=0; k<angle_code.size();k++){//each tilt angle
                vector<Mat> Panning;
                for (int l=0;l<angle_code.size();l++){//each pan angle
                    string temp = Dataset_location + "Person0" + to_string(i+1) + "/person0" + to_string(i+1)+to_string(j) + to_string(m) + angle_code[k]+angle_code[l];
                    if (i>9){
                        temp = Dataset_location + "Person" + to_string(i+1) + "/person" + to_string(i+1)+to_string(j) + to_string(m) + angle_code[k]+angle_code[l];
                    }
                    m++;
                    Panning.push_back(imread(temp+".jpg"));
                    
                    
                    
                    
                    
                }
            }
        
        }
    }
}


///Users/user1/Desktop/uni_archive/semester5/ECSE415vision/project/HeadPoseImageDatabase