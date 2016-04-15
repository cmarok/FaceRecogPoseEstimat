#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "lbp.h"


using namespace cv;
using namespace std;

const int NUM_FOLDS = 7;

void LoadQMUL(vector<vector<vector<Mat>>> &Faces);
void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number, vector<vector<vector<vector<Mat>>>> &Pose);

int main(){
	vector<vector<vector<Mat>>> Faces;
	vector<vector<vector<Mat>>> Pose;
	vector<vector<vector<Mat>>> Hists;
	LoadQMUL(Faces);
	int levels = 0;

	
	/*imshow("hello1", Faces[1][1][0]);
	namedWindow("hello1", WINDOW_AUTOSIZE);
	
	imshow("hello2", Hists[0][0][0]);
	namedWindow("hello2", WINDOW_AUTOSIZE);
	waitKey(0);*/
	int subject_number = 2;

}

void face_recog(int subject, vector<vector<vector<Mat>>> &Faces, int levels){
	vector<vector<vector<Mat>>> Hists;

	//shrink data set
	//keep one set in for each name
	vector<vector<Mat>> sub1;
	int s = 0;
	while (s < 4){
		int i = 0;
		while (i < Faces[s].size()){
			sub1.push_back(Faces[s][i]);
			i++;
		}
		s++;
	}
	vector<vector<vector<Mat>>> sub;
	sub.push_back(sub1);
	//add in sets to test here
	sub.push_back(sub1);
		
	vector<vector<vector<Mat>>> LBP_faces = FacesLBP(sub);
	vector<double>dist(levels);
		//computes spatial pyramid histagram for given images
		for (int k = 0; k < LBP_faces.size(); k++){
			vector<vector<Mat>> Tilting;
			for (int l = 0; l < LBP_faces[k].size(); l++){
				vector<Mat> Panning;
				for (int m = 0; m < LBP_faces[k][l].size(); m++){
					Mat tmp = LBP_faces[k][l][m];
					Mat abc = LBPHistograms(tmp, levels);
					Panning.push_back(abc);
				}
				Tilting.push_back(Panning);
			}
			Hists.push_back(Tilting);
		}
		//get hist for test
		vector<vector<vector<Mat>>> testHists;
		vector<vector<vector<Mat>>> test;
		test.push_back(Faces[subject]);

		for (int k = 0; k < test.size(); k++){
			vector<vector<Mat>> Tilting;
			for (int l = 0; l < test[k].size(); l++){
				vector<Mat> Panning;
				for (int m = 0; m < test[k][l].size(); m++){
					Mat tmp = test[k][l][m];
					Mat abc = LBPHistograms(tmp, levels);
					Panning.push_back(abc);
				}
				Tilting.push_back(Panning);
			}
			testHists.push_back(Tilting);
		}

		double best_score_3;
		int guess_3;
		for (int i = 0; i < Hists.size(); i++){
			for (int j = 0; j < Hists[i].size(); j++){
				for (int k = 0; k < Hists[i][j].size(); k++)
				{
					dist[levels] = compareHist(testHists[i][j][k], Hists[i][j][k], CV_COMP_CHISQR);
					double sum = 0;
					for (int s = 1; s < levels; s++){
						sum = sum + dist[s] / (pow(2, (levels - 1 - s + 1)));
					}
					double diff = dist[0] / (pow(2, (levels - 1))) + sum;

					if (diff < best_score_3){
						best_score_3 = diff;
						guess_3 = i;
					}
				}			
			}
		}

}




void LoadQMUL(vector<vector<vector<Mat>>> &Faces)
{

	//change to location of grey dataset when running on another machine
	const string Dataset_location = "QMUL/";

	const vector<string> Subject_names = { "AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ", "DennisP", "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack", "JamieS", "JeffN", "John", "Jon", "KateS", "KatherineW", "KeithC", "KrystynaN", "PaulV", "RichardB", "RichardH", "SarahL", "SeanG", "SeanGNoGlasses", "SimonB", "SueW", "TasosH", "TomK", "YogeshR", "YongminY" };
	const vector<string> Subject_names2 = { "AdamB", "AndreeaV", "CarlaB", "ColinP", "DanJ", "DennisP", "DennisPNoGlasses", "DerekC", "GrahamW", "HeatherL", "Jack", "JamieS", "JeffNG", "John", "OngEJ", "KateS", "KatherineW", "KeithC", "KrystynaN", "PaulV", "RichardB", "RichardH", "SarahL", "SeanG", "SeanGNoGlasses", "SimonB", "SueW", "TasosH", "TomK", "YogeshR", "YongminY" };
	const vector<string> Tilt_code = { "060", "070", "080", "090", "100", "110", "120" };
	const vector<string> Pan_code = { "000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100", "110", "120", "130", "140", "150", "160", "170", "180" };
	//there may be nicer ways of doing it but this was the fastest to code (and i expect run since no int->str conversions etc)
	//loops load all the images
	for (int i = 0; i < Subject_names.size(); i++){
		vector<vector<Mat>> Tilting;
		for (int j = 0; j<Tilt_code.size(); j++){
			vector<Mat> Panning;
			for (int k = 0; k<Pan_code.size(); k++){
				string temp = Dataset_location + Subject_names[i] + "Grey/" + Subject_names2[i] + "_" + Tilt_code[j] + "_" + Pan_code[k] + ".ras";

				////////////////////////////////
				Panning.push_back(imread(temp));
				////////////////////////////////
				//std::cout << "ssd" << std::endl;

			}
			Tilting.push_back(Panning);
		}

		Faces.push_back(Tilting);
	}
}

void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number)
{
	Mat Display_img;
	for (int j = 0; j<7; j++){
		Mat Row = Faces[Subject_number][j][0];

		for (int k = 1; k<19; k++){
			Mat red = Faces[Subject_number][j][k];
			Mat Row2;
			hconcat(Row, red, Row2);
			Row = Row2;

		}

		if (j == 0){
			Display_img = Row;
		}
		else{
			vconcat(Display_img, Row, Display_img);
		}

	}

	imshow("hello", Display_img);
	namedWindow("hello", WINDOW_AUTOSIZE);
	waitKey(0);

}