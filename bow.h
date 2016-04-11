#ifndef BOW_H
#define BOW_H

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

class BOW
{
private:
    static int const duplicate = 5; // Subjects that have duplicate number of categories (e.g. withglass withoutglass etc)

public:
//    BOW();
    static void faceRecognition(vector<vector<vector<Mat>>> &Faces, Mat &codeBook, vector<vector<vector<Mat>>> &BOWrepresentation, int const numCodeWords);
};

#endif // BOW_H
