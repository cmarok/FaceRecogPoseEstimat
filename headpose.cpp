#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include "headpose.h"

using namespace std;
using namespace cv;

HeadPose::HeadPose(string path)
{
    string dir, file_read;
    int num_image, line_num, x_center, y_center;

    // for each person
    for (int i = 0; i < id.size(); i++) {
        vector<vector<vector<Mat>>> id_vector;
        vector<vector<vector<Rect>>> id_annotation;

        // for each serie
        for (int s = 0; s < serie.size(); s++) {
            vector<vector<Mat>> serie_vector;
            vector<vector<Rect>> serie_annotation;
            num_image = 14;

            // for each tilt from -30 to +30
            for (int t = 0; t < tilt.size(); t++) {
                vector<Mat> tilt_vector;
                vector<Rect> tilt_annotation;

                // for each pan
                for (int p = 0; p < pan.size(); p++) {
                    dir = path + id[i] + "/" + id[i] + serie[s] + to_string(num_image) + tilt[t] + pan[p];
                    num_image++;

                    ifstream file(dir + ".txt");
                    line_num = 0;
                    while (getline(file, file_read))
                    {
                        if (line_num == 3) { x_center = stoi(file_read); }
                        if (line_num == 4) { y_center = stoi(file_read); }
                        line_num++;
                    }

                    cout << "Reading image from " + dir << endl;
                    Mat original = imread(dir + ".jpg");

                    // Check if x_center + 50 or x_center - 50 is still within the boundary of the original image.
                    if (x_center+50 >= original.cols)
                        x_center = original.cols - 50 - 1;
                    if (x_center-50 < 0)
                        x_center = 50;
                    // Check for y_center
                    if (y_center+50 >= original.rows)
                        y_center = original.rows - 50 - 1;
                    if (y_center-50 < 0)
                        y_center = 50;

                    // Store annotations Rect(x, y, width, height)
                    tilt_annotation.push_back(Rect(x_center-50, y_center-50, 100, 100));
                    tilt_vector.push_back(roi);
                }

                serie_annotation.push_back(tilt_annotation);
                serie_vector.push_back(tilt_vector);
            }

            id_annotation.push_back(serie_annotation);
            id_vector.push_back(serie_vector);
        }

        annotations.push_back(id_annotation);
        images.push_back(id_vector);
    }
}

void HeadPose::displayImages(int serie_id, int subject_id)
{
    Mat display;
    Size size(1440, 900);

    for (int t = 0; t < images[subject_id-1][serie_id-1].size(); t++) {
        Mat img_row0 = images[subject_id-1][serie_id-1][t][0];
        rectangle(img_row0, annotations[subject_id-1][serie_id-1][t][0], Scalar(0, 255, 0), 1, 8);

        for (int p = 1; p < images[subject_id-1][serie_id-1][t].size(); p++) {
            Mat img_row = images[subject_id-1][serie_id-1][t][p];
            rectangle(img_row, annotations[subject_id-1][serie_id-1][t][p], Scalar(0, 255, 0), 1, 8);

            hconcat(img_row0, img_row, img_row0);
        }

        // If it is the first row
        if (t == 0) {
            display = img_row0;
        } else {
            vconcat(display, img_row0, display);
        }
    }

    resize(display, display, size);
    imshow("Face Image", display);
    namedWindow("Face Image", WINDOW_AUTOSIZE);
    waitKey(0);
}