#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <math.h>
#include "bow.h"

#define K 7

using namespace std;
using namespace cv;

void BOW::crossValidation(vector<vector<vector<Mat>>> &Faces, int const numCodeWords)
{
    // Shuffle each images in each subject.
    for (int i = 0; i < Faces.size(); i++) {
        random_shuffle(Faces[i].begin(), Faces[i].end());
        for (int j = 0; j < Faces[i].size(); j++) {
            random_shuffle(Faces[i][j].begin(), Faces[i][j].end());
        }
    }

    double ratio = 0.0;

    for (int i = 0; i < K; i++) {
        Mat codeBook;
        vector<vector<vector<Mat>>> faceDescriptors;
        double result_ratio;

        faceRecognition(Faces, codeBook, faceDescriptors, numCodeWords, i);
        result_ratio = faceTest(Faces, codeBook, faceDescriptors, i);
        ratio += result_ratio;
    }

    cout << endl << endl << "Recognition rate for " << numCodeWords << " codewords is " << ratio / K << "." << endl << endl;
}

void BOW::crossValidationProb(vector<vector<vector<Mat>>> &Faces, int const numCodeWords)
{
    // Shuffle each images in each subject.
    for (int i = 0; i < Faces.size(); i++) {
        random_shuffle(Faces[i].begin(), Faces[i].end());
        for (int j = 0; j < Faces[i].size(); j++) {
            random_shuffle(Faces[i][j].begin(), Faces[i][j].end());
        }
    }

    double ratio = 0.0;

    for (int i = 0; i < K; i++) {
        Mat codeBook;
        vector<vector<vector<Mat>>> faceDescriptors;
        vector<Mat> covar, mean;
        double result_ratio;

        faceRecognitionProb(Faces, codeBook, faceDescriptors, numCodeWords, i, mean, covar);
        result_ratio = faceTestProb(Faces, codeBook, faceDescriptors, i, mean, covar);
        ratio += result_ratio;
    }

    cout << endl << endl << "Recognition rate for " << numCodeWords << " codewords is " << ratio / K << "." << endl << endl;
}

void BOW::faceRecognition(vector<vector<vector<Mat>>> &Faces, Mat &codeBook, vector<vector<vector<Mat>>> &BOWrepresentation,
                          int const numCodeWords, int const k_th)
{
    cout << "faceRecognition" << endl;
    // Create SIFT feature detector object & SIFT descriptor extractor object
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    // Create Mat object to store all the SIFT descriptors of all training images of all categories
    Mat D;
    // Create keypoint object
    vector<KeyPoint> keypoints;

    for (int s = 0; s < Faces.size(); s++) {
        for (int t = 0; t < Faces[s].size(); t++) {
            int sub_size = Faces[s][t].size() / K;
            int sub_start = k_th * sub_size;

            for (int p = 0; p < Faces[s][t].size(); p++) {
                if (sub_start <= p && p < sub_start+sub_size) { ; }
                else {
                    detector->detect(Faces[s][t][p], keypoints);

                    Mat tmp;
                    // Comute the SIFT descriptor for the keypoints.
                    extractor->compute(Faces[s][t][p], keypoints, tmp);
                    // Add the descriptors of the current image to D.
                    D.push_back(tmp);
                }
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

    for (int s = 0; s < Faces.size(); s++) {
        vector<vector<Mat>> hist_sub;

        for (int t = 0; t < Faces[s].size(); t++) {
            int sub_size = Faces[s][t].size() / K;
            int sub_start = k_th * sub_size;
            vector<Mat> hist_tilt;

            for (int p = 0; p < Faces[s][t].size(); p++) {
                if (sub_start <= p && p < sub_start+sub_size) { ; }
                else {
                    detector->detect(Faces[s][t][p], keypoints);

                    // Compute the bag of histogram representation.
                    Mat hist;
                    bowExtractor->compute2(Faces[s][t][p], keypoints, hist);
                    normalize(hist, hist, 0, 100, NORM_MINMAX, -1, Mat());

                    hist_tilt.push_back(hist);
                }
            }

            hist_sub.push_back(hist_tilt);
        }
        BOWrepresentation.push_back(hist_sub);
    }
}

double BOW::faceTest(const vector<vector<vector<Mat>>> &Faces, const Mat &codeBook,
                     const vector<vector<vector<Mat>>> &BOWrepresentation, int const k_th)
{
    cout << "faceTest" << endl;
    // Create SIFT feature detector object & SIFT descriptor extractor object
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowExtractor = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor, matcher));

    // Create keypoint object
    vector<KeyPoint> keypoints;
    // Set the codebook of the bag of words descriptor extractor object.
    bowExtractor->setVocabulary(codeBook);

    double min;
    int index_result, correct, total_test;
    correct = 0;
    total_test = 0;

    for (int s = 0; s < Faces.size(); s++) {
            for (int t = 0; t < Faces[s].size(); t++) {
                int sub_size = Faces[s][t].size() / K;
                int sub_start = k_th * sub_size;

                for (int p = sub_start; p < sub_start+sub_size; p++) {
                    if (!Faces[s][t][p].empty()) {
                        // Store the total number of test images to be used to calculate ratio.
                        total_test++;

                        // Detect SIFT key points in image.
                        detector->detect(Faces[s][t][p], keypoints);

                        Mat descriptor;
                        bowExtractor->compute2(Faces[s][t][p], keypoints, descriptor);

                        // normalize descriptor
                        normalize(descriptor, descriptor, 0, 100, NORM_MINMAX, -1, Mat());

                        min = numeric_limits<double>::max();

                        // Calculate the distance.
                        for (int s1 = 0; s1 < BOWrepresentation.size(); s1++) {
                            for (int t1 = 0; t1 < BOWrepresentation[s1].size(); t1++) {
                                for (int p1 = 0; p1 < BOWrepresentation[s1][t1].size(); p1++) {
                                    // Calculate the chi square distance
                                    Mat a = descriptor - BOWrepresentation[s1][t1][p1];
                                    Mat b = a.mul(a);
                                    Mat c = b / (descriptor + BOWrepresentation[s1][t1][p1]);

                                    double result = sum(c)[0];

                                    if (result < min) {
                                        min = result;
                                        index_result = s1;
                                    }
                                }
                            }
                        }

                        if (s == index_result) {
                            correct++;
                        }
                        else {
//                            cout << "Failed to find correct result for Faces[" << s<< "][" << t<< "][" << p << "]." << endl;
                        }
                    }
                }
            }

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

    double result_ratio = (double)correct / (double)(total_test);
    cout << "At Codeword " << numCodeWords << "@k = " << k_th << " recognition rate: " << result_ratio << endl;

    return result_ratio;
}

void BOW::faceRecognitionProb(vector<vector<vector<Mat>>> &Faces, Mat &codeBook,
                              vector<vector<vector<Mat>>> &BOWrepresentation, int const numCodeWords, int const k_th,
                              vector<Mat> &mean, vector<Mat> &covar)
{
    // Calculate codeBook and BOWrepresentation.
    faceRecognition(Faces, codeBook, BOWrepresentation, numCodeWords, k_th);

    // To calculate Gaussian distribution, mean matrix and covariance matrix needed.
    // Calculate covar and mean for each subject id.
    for (int i = 0; i < BOWrepresentation.size(); i++) {
        vector<Mat> sample;
        Mat covar_sample, mean_sample;

        for (int j = 0; j < BOWrepresentation[i].size; j++) {
            for (int l = 0; l < BOWrepresentation[i][j].size; l++) {
                Mat descriptor = BOWrepresentation[i][j][l];

                // Format change
                descriptor.convertTo(descriptor, CV_64F);
                sample.push_back(descriptor);
            }
        }

        calcCovarMatrix(descriptor, covar_sample, mean_sample,CV_COVAR_NORMAL);
        mean.push_back(mean_sample);
        covar.push_back(covar_sample);
    }
}

double BOW::faceTestProb(const vector<vector<vector<Mat>>> &Faces, const Mat &codeBook,
                     const vector<vector<vector<Mat>>> &BOWrepresentation, int const k_th,
                         const vector<Mat> &mean, const vector<Mat> &covar)
{
    // Create SIFT feature detector object & SIFT descriptor extractor object
    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowExtractor = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor, matcher));

    // Create keypoint object
    vector<KeyPoint> keypoints;
    // Set the codebook of the bag of words descriptor extractor object.
    bowExtractor->setVocabulary(codeBook);

    double max;
    int index_result, correct, total_test;
    correct = 0;
    total_test = 0;

    for (int s = 0; s < Faces.size(); s++) {
            for (int t = 0; t < Faces[s].size(); t++) {
                int sub_size = Faces[s][t].size() / K;
                int sub_start = k_th * sub_size;

                for (int p = sub_start; p < sub_start+sub_size; p++) {
                    if (!Faces[s][t][p].empty()) {
                        // Store the total number of test images to be used to calculate ratio.
                        total_test++;

                        // Detect SIFT key points in image.
                        detector->detect(Faces[s][t][p], keypoints);

                        Mat descriptor;
                        bowExtractor->compute2(Faces[s][t][p], keypoints, descriptor);

                        // normalize descriptor
                        normalize(descriptor, descriptor, 0, 100, NORM_MINMAX, -1, Mat());

                        // convert descriptor to CV_64F to match covar & mean
                        descriptor.convertTo(descriptor, CV_64F);
                        int k_gaussian = descriptor.cols;

                        max = 0.0;

                        for (int i = 0; i < mean.size(); i++) {
                            Mat difference = descriptor - mean[i];
                            Mat exponent = (difference.t() * covar[i].inv() * difference) / -2;
                            Mat result = exp(exponent) / sqrt(pow(2 * M_PI, k_gaussian) * determinant(covar[i]));

                            double prob = sum(result)[0];

                            if (prob > max) {
                                max = prob;
                                index_result = i;
                            }
                        }

                        if (s == index_result) {
                            correct++;
                        }
                        else {
//                            cout << "Failed to find correct result for Faces[" << s<< "][" << t<< "][" << p << "]." << endl;
                        }
                    }
                }
            }

    }

    double result_ratio = (double)correct / (double)(total_test);
    cout << "At Codeword " << numCodeWords << "@k = " << k_th << " recognition rate: " << result_ratio << endl;

    return result_ratio;
}
