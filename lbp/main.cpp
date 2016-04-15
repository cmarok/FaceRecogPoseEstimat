
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
#include "stdafx.h"
#include "lbp.h"


using namespace cv;
using namespace std;

const int NUM_FOLDS = 7;

int face_recog(int subject, vector<vector<vector<Mat>>> &Faces, int levels);
vector<vector<vector<Mat>>> spatialPyramid(vector<vector<vector<Mat>>>&faces, int levels);
void LoadQMUL(vector<vector<vector<Mat>>> &Faces);
void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number, vector<vector<vector<vector<Mat>>>> &Pose);

int main(){
	vector<vector<vector<Mat>>> Faces;
	vector<vector<vector<Mat>>> Pose;
	vector<vector<vector<Mat>>> Hists;
	//LoadQMUL(Faces);
	int levels = 0;

	
	/*imshow("hello1", Faces[1][1][0]);
	namedWindow("hello1", WINDOW_AUTOSIZE);
	
	imshow("hello2", Hists[0][0][0]);
	namedWindow("hello2", WINDOW_AUTOSIZE);
	waitKey(0);*/
	int subject_number = 2;
	int result = face_recog(subject_number, Faces, 3);

}

vector<vector<vector<Mat>>> spatialPyramid(vector<vector<vector<Mat>>>&faces, int levels){
	vector<vector<vector<Mat>>> Hists;
	for (int k = 0; k < faces.size(); k++){
		vector<vector<Mat>> Tilting;
		for (int l = 0; l < faces[k].size(); l++){
			vector<Mat> Panning;
			for (int m = 0; m < faces[k][l].size(); m++){
				Mat tmp = faces[k][l][m];
				Mat abc = LBPHistograms(tmp, levels);
				Panning.push_back(abc);
			}
			Tilting.push_back(Panning);
		}
		Hists.push_back(Tilting);
	}
	return Hists;
}
int face_recog(int subject, vector<vector<vector<Mat>>> &Faces, int levels){
	
	vector<vector<vector<Mat>>> folded = Faces;
	/*folded.resize(7);
	for (int i = 0; i < folded.size(); i++){
		srand(31);
		seven_fold_cv
	}*/
	//shrink data set

	//then load
	//scramble data
	for (int i = 0; i < folded.size(); i++) {
		random_shuffle(folded[i].begin(), folded[i].end());
		for (int j = 0; j < folded[i].size(); j++) {
			random_shuffle(folded[i][j].begin(), folded[i][j].end());
		}
	}
	vector<Mat > sub2;
	for (int qwerty = 0; qwerty < folded.size(); qwerty++){
		sub2.push_back(folded[qwerty][1][1]);

	}

	vector<vector<Mat>> sub1;
	sub1.push_back(sub2);
	vector<vector<vector<Mat>>> sub;
	sub.push_back(sub1);
	//vector<Mat > sub2;
	//for (int qwerty = 0; qwerty < folded.size(); qwerty++){
	//	sub2.push_back(folded[qwerty][srand(6)][srand(18)]);

	//}
	

	//int s = 0;
	//while (s < 4){
	//	int i = 0;
	//	while (i < folded[s].size()){
	//		sub1.push_back(folded[s][i]);
	//		i++;
	//	}
	//	s++;
	//}
	//vector<vector<vector<Mat>>> sub;
	//sub.push_back(sub1);
	////add in sets to test here
	//sub.push_back(sub1);

		
	vector<vector<vector<Mat>>> LBP_faces = FacesLBP(sub);
	//0level hists
	vector<vector<vector<Mat>>> hist0 = spatialPyramid(LBP_faces, 0);
	//0level person
	vector<vector<vector<Mat>>> testHists;
	vector<vector<vector<Mat>>> test;
	test.push_back(folded[subject]);

	vector<vector<vector<Mat>>> subj0 = spatialPyramid(test, 0);


	vector<double>dist(levels);
	double best = numeric_limits<double>::max();
	int guess;

	for (int i = 0; i < hist0.size(); i++){
		for (int j = 0; j < hist0[i].size(); j++){
			for (int k = 0; k < hist0[i][j].size(); k++)
			{
				dist[0] = compareHist(subj0[i][j][k], hist0[i][j][k], CV_COMP_CHISQR);

				double diff = dist[0];
				if (diff < best) {
					best = diff;
					guess = k;
				}
			}
		}
	}

	//1 level

	vector<vector<vector<Mat>>> hist1 = spatialPyramid(LBP_faces, 1);
	//1level person

	vector<vector<vector<Mat>>> subj1 = spatialPyramid(test, 1);


	for (int i = 0; i < hist0.size(); i++){
		for (int j = 0; j < hist0[i].size(); j++){
			for (int k = 0; k < hist0[i][j].size(); k++)
			{
				dist[1] = compareHist(subj1[i][j][k], hist1[i][j][k], CV_COMP_CHISQR);

				double sum = 0;
				sum = (sum + dist[1])/2;

				double diff = dist[0] / 2 + sum;
				if (diff < best) {
					best = diff;
					guess = k;
				}
			}
		}
	}

	//2 level

	vector<vector<vector<Mat>>> hist2 = spatialPyramid(LBP_faces, 2);
	//2level person

	vector<vector<vector<Mat>>> subj2 = spatialPyramid(test, 2);

	for (int i = 0; i < hist0.size(); i++){
		for (int j = 0; j < hist0[i].size(); j++){
			for (int k = 0; k < hist0[i][j].size(); k++)
			{
				dist[2] = compareHist(subj2[i][j][k], hist2[i][j][k], CV_COMP_CHISQR);

				double sum = 0;
				for (int lvl = 0; lvl < 2; lvl++){
					sum = (sum + dist[lvl]) / (pow(2, (2 - lvl + 1)));
				}
				

				double diff = dist[0] / (pow(2, (2))) + sum;
				if (diff < best) {
					best = diff;
					guess = k;
				}
			}
		}
	}

	//3 level

	vector<vector<vector<Mat>>> hist3 = spatialPyramid(LBP_faces, 3);
	//3level person

	vector<vector<vector<Mat>>> subj3 = spatialPyramid(test, 3);


	for (int i = 0; i < hist0.size(); i++){
		for (int j = 0; j < hist0[i].size(); j++){
			for (int k = 0; k < hist0[i][j].size(); k++)
			{
				dist[1] = compareHist(subj3[i][j][k], hist3[i][j][k], CV_COMP_CHISQR);

				double sum = 0;
				for (int lvl = 0; lvl < 3; lvl++){
					sum = (sum + dist[lvl]) / (pow(2, (3 - lvl + 1)));
				}


				double diff = dist[0] / (pow(2, (3))) + sum;
				if (diff < best) {
					best = diff;
					guess = k;
				}
			}
		}
	}
	return guess;
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