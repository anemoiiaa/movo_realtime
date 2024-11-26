#include "../include/movo_realtime/vo_features.hpp"
#include "../include/movo_realtime/movo_realtime.hpp"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)	{

  string line;
  int i = 0;
  ifstream myfile("/home/robit/data_odometry_gray/dataset/sequences/00/times.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }

      i++;
    }
    myfile.close();
  }

  else {
    cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}

void MovoRealtime::image_processing(const cv::Mat& img)
{
  static bool first = true;
  static bool second = false;

  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);

  if(first && !second)
  {
    currImage = img.clone();
    second = true;
    first = false;
    return;
  }
  else if(second && !first)
  {
    prevImage = currImage.clone();
    currImage = img.clone();


    Mat img_1, img_2;

    cvtColor(prevImage, img_1, COLOR_BGR2GRAY);
    cvtColor(currImage, img_2, COLOR_BGR2GRAY);

    featureDetection(img_1, keypoints_1, descriptors_1);
    featureTracking(img_1, img_2, keypoints_1, keypoints_2, descriptors_1, descriptors_2, good_matches);

    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    E = findEssentialMat(points2, points1, focalLen.x(), prncPt, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focalLen.x(), prncPt, mask);

    R_f = R.clone();
    t_f = t.clone();
  }
  else if(!first && !second)
  {
    prevImage = currImage.clone();
    currImage = img.clone();

    vector<KeyPoint> keypoints_2;
    Mat descriptors_2;
    vector<DMatch> good_matches;

  }
}

MovoRealtime::MovoRealtime() : Node("movo_realtime")
{
  image_topic = this->declare_parameter<std::string>("image_topic", "/camera/image_raw");

  image_topic = this->get_parameter("image_topic").as_string();

  image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    image_topic, 10, std::bind(&MovoRealtime::image_callback, this, std::placeholders::_1));

  info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "/camera/camera_info", 10, [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {

      K_M = cv::Mat(3, 3, CV_64F, (void *)msg->k.data());
      D_M = cv::Mat(1, 5, CV_64F, (void *)msg->d.data());
      R_M = cv::Mat(3, 3, CV_64F, (void *)msg->r.data());
      P_M = cv::Mat(3, 4, CV_64F, (void *)msg->p.data());

    });

  calibration_info();

}

MovoRealtime::~MovoRealtime() {
    // 소멸자 구현
}

void MovoRealtime::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  static cv::Mat frame;  // memory leak
  try {
    frame = cv_bridge::toCvShare(msg, "bgr8")->image;

    if (frame.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Received empty frame.");
      return;
    }



   if (cv::waitKey(10) == 27) {
      rclcpp::shutdown();
    }
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(
      this->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }

  focalLen = cv::Point2d(K_M.at<double>(0, 0), K_M.at<double>(1, 1));
  prncPt = cv::Point2d(K_M.at<double>(0, 2), K_M.at<double>(1, 2));

  image_processing(frame);

}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<MovoRealtime>());

  rclcpp::shutdown();
  return 0;
}
