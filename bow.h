#ifndef BOW_H
#define BOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

class BOW
{
public:
    static void crossValidation(vector<vector<vector<Mat>>> &Faces, int const numCodeWords);

    static void crossValidationProb(vector<vector<vector<Mat>>> &Faces, int const numCodeWords);

    static void faceRecognition(vector<vector<vector<Mat>>> &Faces, Mat &codeBook,
                                vector<vector<vector<Mat>>> &BOWrepresentation, int const numCodeWords, int const k_th);

    static double faceTest(const vector<vector<vector<Mat>>> &Faces, const Mat &codeBook,
                           const vector<vector<vector<Mat>>> &BOWrepresentation, int const k_th);

    static void faceRecognitionProb(vector<vector<vector<Mat>>> &Faces, Mat &codeBook,
                                    vector<vector<vector<Mat>>> &BOWrepresentation, int const numCodeWords, int const k_th,
                                    vector<Mat> &mean, vector<Mat> &covar);

    static double faceTestProb(const vector<vector<vector<Mat>>> &Faces, const Mat &codeBook,
                         const vector<vector<vector<Mat>>> &BOWrepresentation, int const k_th,
                             const vector<Mat> &mean, const vector<Mat> &covar);
};

#endif // BOW_H
