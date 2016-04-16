#ifndef BOW_H
#define BOW_H

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <stdio.h>
#include "headpose.h"

using namespace std;
using namespace cv;

class BOW
{
public:
    static void recognition(vector<vector<vector<Mat>>> &Faces, int const numCodeWords);

    static void recognitionP(vector<vector<vector<Mat>>> &Faces, int const numCodeWords);

    static void faceTrain(vector<vector<vector<Mat>>> &Faces, Mat &codeBook,
                                vector<vector<vector<Mat>>> &BOWrepresentation, int const numCodeWords, int const k_th);

    static double faceTest(const vector<vector<vector<Mat>>> &Faces, const Mat &codeBook,
                           const vector<vector<vector<Mat>>> &BOWrepresentation, int const k_th);

    static void faceTrainP(vector<vector<vector<Mat>>> &Faces, Mat &codeBook,
                                    vector<vector<vector<Mat>>> &BOWrepresentation, int const numCodeWords, int const k_th,
                                    vector<Mat> &mean, vector<Mat> &covar);

    static double faceTestP(const vector<vector<vector<Mat>>> &Faces, const Mat &codeBook,
                         const vector<vector<vector<Mat>>> &BOWrepresentation, int const k_th,
                             const vector<Mat> &mean, const vector<Mat> &covar);

    static void poseRecognition(vector<vector<vector<Mat>>> pose, HeadPose hp);

    static void poseTrain(vector<vector<vector<Mat>>> &pose, Mat &codeBook, vector<vector<vector<Mat>>> &poseDescriptors,
                              int const numCodeWords);

    static void poseTest(const HeadPose hp, const Mat &codeBook, const vector<vector<vector<Mat>>> &poseDescriptor);
};

#endif // BOW_H
