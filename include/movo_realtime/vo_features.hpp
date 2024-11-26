/*

The MIT License

Copyright (c) 2015 Avi Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;



void printMatches(const vector<vector<DMatch>>& matches) {
    for (size_t i = 0; i < matches.size(); i++) {
        cout << "Match " << i << ": ";
        for (size_t j = 0; j < matches[i].size(); j++) {
            cout << "[QueryIdx: " << matches[i][j].queryIdx
                 << ", TrainIdx: " << matches[i][j].trainIdx
                 << ", Distance: " << matches[i][j].distance << "] ";
        }
        cout << endl;
    }
}


void featureDetection(Mat img_1, vector<KeyPoint>& keypoints_1, Mat& descriptors_1) {
    const static auto& orb = cv::ORB::create();
    orb->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptors_1);
}

void featureTracking(Mat img_1, Mat img_2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptors_1, Mat& descriptors_2, vector<DMatch>& good_matches, vector<Point2f>& prevFeatures, vector<Point2f>& currFeatures) {

    
    const static auto& orb = cv::ORB::create();
    orb->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptors_2);

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
            prevFeatures.push_back(keypoints_1[knn_matches[i][0].queryIdx].pt);
            currFeatures.push_back(keypoints_2[knn_matches[i][0].trainIdx].pt);
        }
    }
}
