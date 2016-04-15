// compVisionProject.cpp : Defines the entry point for the console application.
//

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

//void LoadQMUL(vector<vector<vector<Mat>>> &Faces);

//void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number, vector<vector<vector<vector<Mat>>>> &Pose);

double computePixelLBP(const Mat input);


/*int main(int argc, const char * argv[]) {



	//load faces array
	vector<vector<vector<Mat>>> Faces;
	LoadQMUL(Faces);

	//display subject function
	//Display_subject(Faces,35,Pose);

	//convert to LBP images
	vector<vector<vector<Mat>>> LBP_faces = FacesLBP(Faces);

	vector<vector<vector<Mat>>> Histograms = LBPHistograms(LBP_faces, 1);

	//testing - can delete
	imshow("hello1", Faces[12][0][0]);
	namedWindow("hello1", WINDOW_AUTOSIZE);
	waitKey(0);
	imshow("hello2", Histograms[12][0][0]);
	namedWindow("hello2", WINDOW_AUTOSIZE);
	waitKey(0);
	//\testing



	return 0;
}*/

/*
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


void Display_subject(vector<vector<vector<Mat>>> &Faces, int Subject_number, vector<vector<vector<vector<Mat>>>> &Pose)
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

	Mat Display_img2;
	for (int j = 0; j<5; j++){
		Mat Row = Pose[14][1][j][0];

		for (int k = 1; k<13; k++){
			Mat red = Pose[14][1][j][k];
			Mat Row2;
			hconcat(Row, red, Row2);
			Row = Row2;

		}

		if (j == 0){
			Display_img2 = Row;
		}
		else{
			vconcat(Display_img2, Row, Display_img2);
		}

	}

	imshow("hello2", Display_img2);
	namedWindow("hello2", WINDOW_AUTOSIZE);
	waitKey(0);


}

*/
//converts faces to LBP images
vector<vector<vector<Mat>>> FacesLBP(vector<vector<vector<Mat>>> &Faces){
	vector<vector<vector<Mat>>> LBP_faces = Faces;


	//for all of the images in the object
	for (int k = 0; k<Faces.size(); k++){
		for (int l = 0; l<Faces[k].size(); l++){
			for (int m = 0; m<Faces[k][l].size(); m++){

				Mat image1 = Faces[k][l][m];
				Mat image_out(image1.size(), CV_64F);


				//for all of the pixels (not including border pixels)
				for (int i = 1; i < image1.cols - 1; i++){
					for (int j = 1; j < image1.rows - 1; j++){
						//extract surrounding 3x3 matrix for every pixel
						Mat pixelNeighbors = image1(Rect(i - 1, j - 1, 3, 3));

						//comupte lbp
						image_out.at<double>(j, i) = computePixelLBP(pixelNeighbors);

						std::cout << "ssd" << std::endl;
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
	for (int i = 0; i < Binary_vec.size() - 1; i++){
		if (Binary_vec[i + 1] - Binary_vec[i] != 0)
			transitions = transitions + 1;
	}
	if (Binary_vec[0] - Binary_vec[Binary_vec.size() - 1] != 0){
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
Mat LBPHistograms(Mat LBP_face, int level)
{
	if (level == 0)
	{
		Mat patch = LBP_face;
		Mat hist;
		int nbins = 59; // 59 bins
		int histSize[] = { nbins }; //one dimension
		float range[] = { 0, 255 }; //up to 255 value
		const float *ranges[] = { range };
		int channels[] = { 0 };
		calcHist(&patch, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
		//normalize(hist, hist, NORM_L2);
		imshow("hello1", hist);
		return hist;

	}
	else
	{
		int width = LBP_face.cols/2;
		int height = LBP_face.rows/2;
		Mat tl, tr, bl, br, res0, res1, output;
		tl = LBPHistograms(LBP_face(Rect(0, 0, width - 1, height - 1)), level - 1);
		tr = LBPHistograms(LBP_face(Rect(width, 0, width - 1, height - 1)), level - 1);
		bl = LBPHistograms(LBP_face(Rect(0, height, width - 1, height - 1)), level - 1);
		br = LBPHistograms(LBP_face(Rect(width, height, width - 1, height - 1)), level - 1);
		
		hconcat(tl, tr, res0);
		hconcat(bl, br, res1);
		hconcat(res0, res1, output);

		normalize(output, output, NORM_L2);
		return output;
	}
}
