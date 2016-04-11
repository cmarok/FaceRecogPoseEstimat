#include <iostream>
#include <opencv2/opencv.hpp>
#include "bow.h"

using namespace std;
using namespace cv;

//BOW::BOW()
//{

//}

void BOW::faceRecognition(vector<vector<vector<Mat>>> &Faces, Mat &codeBook, vector<vector<vector<Mat>>> &BOWrepresentation,
                          int const numCodeWords)
{
    int numSubject = Faces.size();
//    cout << "Num Subjects: " << numSubject << endl << endl;

    // Create SIFT feature detector object & SIFT descriptor extractor object
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    // Create Mat object to store all the SIFT descriptors of all training images of all categories
    Mat D;
    // Create keypoint object
    vector<KeyPoint> keypoints;

    for (int s = 0; s < numSubject; s++) {
        int numTilt = Faces[s].size();
//        cout << "Num Tilt for Faces[" << s << "]: " << numTilt << endl;

        for (int t = 0; t < numTilt; t++) {
            int numPan = Faces[s][t].size();
//            cout << "Num Tilt for Faces[" << s << "][" << t << "]: " << numPan << endl;

            for (int p = 0; p < numPan; p++) {
                // Detect SIFT key points in image.
                detector->detect(Faces[s][t][p], keypoints);

                Mat tmp;
                // Comute the SIFT descriptor for the keypoints.
                extractor->compute(Faces[s][t][p], keypoints, tmp);
                // Add the descriptors of the current image to D.
                D.push_back(tmp);

//                Mat outputImage;
//                drawKeypoints(Faces[s][t][p], keypoints, outputImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//                namedWindow("Output");
//                imshow("Output", outputImage);
//                waitKey(0);

                cout << "SIFT extracted for Faces[" << s << "][" << t << "][" << p << "]" << endl;
            }
        }
    }

    // Create a bag of words trainer object.
    BOWKMeansTrainer bow(numCodeWords, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, FLT_EPSILON), 1, KMEANS_PP_CENTERS);

    // Add the descriptors to the bag of words trainer object.
    bow.add(D);

    // Compute the codebook.
    codeBook = bow.cluster();

    // Create a Brute Force descriptor matcher object & a bag of words descriptor extractor object.
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowExtractor = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor, matcher));

    // Set the codebook of the bag of words descriptor extractor object.
    bowExtractor->setVocabulary(codeBook);

//    cout << "Dimensions of CodeBook ----- H: " << codeBook.rows << " W: " << codeBook.cols << endl;
    for (int s = 0; s < numSubject; s++) {
        int numTilt = Faces[s].size();
        vector<vector<Mat>> hist_sub;

        for (int t = 0; t < numTilt; t++) {
            int numPan = Faces[s][t].size();
            vector<Mat> hist_tilt;

            for (int p = 0; p < numPan; p++) {
                detector->detect(Faces[s][t][p], keypoints);

                // Compute the bag of histogram representation.
                Mat hist;
                bowExtractor->compute2(Faces[s][t][p], keypoints, hist);
                normalize(hist, hist, 0, 100, NORM_MINMAX, -1, Mat());

                hist_tilt.push_back(hist);

                cout << "Histogram created for Faces[" << s << "][" << t << "][" << p << "]" << endl;
            }

            hist_sub.push_back(hist_tilt);
        }
        BOWrepresentation.push_back(hist_sub);
    }

//    // Draw the histograms
//      int hist_w = 100; int hist_h = 100;
//      int bin_w = cvRound( (double) hist_w/numCodeWords );

//      for (int p = 1; p < Faces[0][0].size(); p++) {
//          // Draw for each channel
//          Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

//          for(int i = 1; i < numCodeWords; i++ ) {
//              line(histImage, Point(bin_w*(i-1), hist_h - cvRound(BOWrepresentation[0][0][p].at<float>(i-1))),
//                      Point(bin_w*(i), hist_h - cvRound(BOWrepresentation[0][0][p].at<float>(i))),
//                      Scalar(255, 255, 255), 2, 8, 0);
//          }

//          // Display
//          namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
//          imshow("calcHist Demo", histImage );
//          waitKey(0);
//      }
}
