#ifndef HEADPOSE_H
#define HEADPOSE_H

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

class HeadPose
{
private:
    const vector<string> id = {"Person01", "Person02", "Person03", "Person04", "Person05", "Person06", "Person07",
                               "Person08", "Person09", "Person10", "Person11", "Person12", "Person13", "Person14",
                               "Person15"};

    const vector<string> serie = {"1", "2"};

    const vector<string> tilt = {"-30", "-15", "+0", "+15", "+30"};

    const vector<string> pan = {"-90", "-75", "-60", "-45", "-30", "-15", "+0", "+15", "+30", "+45", "+60", "+75", "+90"};

public:
    vector<vector<vector<vector<Mat>>>> images;

    vector<vector<vector<vector<Rect>>>> annotations;

    HeadPose(string path);

    void displayImages(int serie, int subject);
};

#endif // HEADPOSE_H
