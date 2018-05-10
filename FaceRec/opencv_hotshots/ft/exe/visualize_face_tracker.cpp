/*
  visualize_face_tracker: perform face tracking from a video/camera stream
*/
#include "opencv_hotshots/ft/svm.h"
#include "opencv_hotshots/ft/ft.h"
#include <opencv2/highgui/highgui.hpp>
#include "ml.h"
#include <iostream>
#define fl at<float>
const char* usage = "usage: ./visualise_face_tracker tracker [video_file]";

vector<Point2f> cPoints;      //an array that stores characteristic points
float featureScaler;       //used to calculate local feature
CvSVM svm;
cv::Point textArea;

float innerBrowRaiser;     //AU 1
float outerBrowRaiser;     //AU 2
float browLower;           //AU 4
float upperLidRaiser;      //AU 5
float lidTightener;        //AU 7
float noseWrinkler;        //AU 9
float lipCornerPull;       //AU 12
float lipCornerDepress;    //AU 15
float lowerLipDepress;     //AU 16
float lipStretch;          //AU 20
float lipTightener;        //AU 23
float jawDrop;             //AU 26

float getDist(cv::Point p1, cv::Point p2)
{
	float r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	return r;
}

float getDistX(cv::Point p1, cv::Point p2)
{
	float r = abs(p1.x - p2.x);
	return r;
}

float getDistY(cv::Point p1, cv::Point p2)
{
	float r = abs(p1.y - p2.y);
	return r;
}

//==============================================================================
void
draw_string(Mat img,                       //image to draw on
        const string text)             //text to draw
{
  Size size = getTextSize(text,FONT_HERSHEY_COMPLEX,0.6f,1,NULL);
  putText(img,text,Point(0,size.height),FONT_HERSHEY_COMPLEX,0.6f,
      Scalar::all(0),1,CV_AA);
  putText(img,text,Point(1,size.height+1),FONT_HERSHEY_COMPLEX,0.6f,
      Scalar::all(255),1,CV_AA);
}
//==============================================================================
bool
parse_help(int argc,char** argv)
{
  for(int i = 1; i < argc; i++){
    string str = argv[i];
    if(str.length() == 2){if(strcmp(str.c_str(),"-h") == 0)return true;}
    if(str.length() == 6){if(strcmp(str.c_str(),"--help") == 0)return true;}
  }return false;
}
//==============================================================================
int main(int argc,char** argv)
{
  //parse command line arguments
  if(parse_help(argc,argv)){cout << usage << endl; return 0;}
  if(argc < 2){cout << usage << endl; return 0;}
  
  //load detector model
  face_tracker tracker = load_ft<face_tracker>(argv[1]);

  //create tracker parameters
  face_tracker_params p; p.robust = false;
  p.ssize.resize(3);
  p.ssize[0] = Size(21,21);
  p.ssize[1] = Size(11,11);
  p.ssize[2] = Size(5,5);

  //open video stream
  VideoCapture cam; 
  if(argc > 2)cam.open(argv[2]); else cam.open(0);
  if(!cam.isOpened()){
    cout << "Failed opening video file." << endl
     << usage << endl; return 0;
  }
  //load svm model
  svm.load("emotion.yaml");

  //detect until user quits
  namedWindow("face tracker");
  while(cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999){
    Mat im; cam >> im; 
	if (tracker.track(im, p)) {
		tracker.draw(im);

		cPoints = tracker.points;
		//assign feature scaler as the width of the face, which does not change in response to different expression
		featureScaler = (getDistX(cPoints[0], cPoints[16]) + getDistX(cPoints[1], cPoints[15]) + getDistX(cPoints[2], cPoints[14])) / 3;
		//assign action unit 1
		innerBrowRaiser = ((getDistY(cPoints[21], cPoints[27]) + getDistY(cPoints[22], cPoints[27])) / 2) / featureScaler;
		//assign action unit 2
		outerBrowRaiser = ((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[26], cPoints[27])) / 2) / featureScaler;
		//assign action unit 4
		browLower = (((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[18], cPoints[27]) +
			getDistY(cPoints[19], cPoints[27]) + getDistY(cPoints[20], cPoints[27]) +
			getDistY(cPoints[21], cPoints[27])) / 5 +
			(getDistY(cPoints[22], cPoints[27]) + getDistY(cPoints[23], cPoints[27]) +
			getDistY(cPoints[24], cPoints[27]) + getDistY(cPoints[25], cPoints[27]) +
			getDistY(cPoints[26], cPoints[27])) / 5) / 2) / featureScaler;
		//assign action unit 5
		upperLidRaiser = ((getDistY(cPoints[37], cPoints[27]) + getDistY(cPoints[44], cPoints[27])) / 2) / featureScaler;
		//assign action unit 7
		lidTightener = ((getDistY(cPoints[37], cPoints[41]) + getDistY(cPoints[38], cPoints[40])) / 2 +
			(getDistY(cPoints[43], cPoints[47]) + getDistY(cPoints[44], cPoints[46])) / 2) / featureScaler;
		//assign action unit 9
		noseWrinkler = (getDistY(cPoints[29], cPoints[27]) + getDistY(cPoints[30], cPoints[27])) / featureScaler;
		//assign action unit 12
		lipCornerPull = ((getDistY(cPoints[48], cPoints[33]) + getDistY(cPoints[54], cPoints[33])) / 2) / featureScaler;
		//assign action unit 16
		lowerLipDepress = getDistY(cPoints[57], cPoints[33]) / featureScaler;
		//assign action unit 20
		lipStretch = getDistX(cPoints[48], cPoints[54]) / featureScaler;
		//assign action unit 23
		lipTightener = (getDistY(cPoints[49], cPoints[59]) +
			getDistY(cPoints[50], cPoints[58]) +
			getDistY(cPoints[51], cPoints[57]) +
			getDistY(cPoints[52], cPoints[56]) +
			getDistY(cPoints[53], cPoints[55])) / featureScaler;
		//assign action unit 26
		jawDrop = getDistY(cPoints[8], cPoints[27]) / featureScaler;

		float class_nr = 0;
		int class_nr_int = 0;
		int i = 0;

		//predict
		float node[11];

		//assign value of nodes
		node[0] = innerBrowRaiser;
		node[1] = outerBrowRaiser;
		node[2] = browLower;
		node[3] = upperLidRaiser;
		node[4] = lidTightener;
		node[5] = noseWrinkler;
		node[6] = lipCornerPull;
		node[7] = lowerLipDepress;
		node[8] = lipStretch;
		node[9] = lipTightener;
		node[10] = jawDrop;

		Mat DataMat(11, 1, CV_32FC1, node);

		//predict the class
		//0: neutral face
		//1: happy
		//2: angry
		//3: disgust
		//-1 sad
		//-2 suprise
		//-3 fear
		class_nr_int = (int)svm.predict(DataMat);

		string notification;
		textArea = cv::Point(cPoints[8].x - 50, cPoints[8].y + 50);
		if (class_nr_int == 2) {
			notification = "Angry";
		}
		else if (class_nr_int == 1){
			notification = "Happy";
		}
		else if (class_nr_int == 3){
			notification = "Disgust";
		}
		else if (class_nr_int == -1){
			notification = "Sad";
		}
		else if (class_nr_int == -2){
			notification = "Surprise";
		}
		else if (class_nr_int == -3){
			notification = "Fear";
		}
		else
		{
			notification = "Neutral";
		}
		Size size = getTextSize(notification, FONT_HERSHEY_COMPLEX, 0.6f, 1, NULL);
		putText(im, notification, Point(0, 3*size.height), FONT_HERSHEY_COMPLEX, 0.6f,
			Scalar::all(0), 1, CV_AA);
		putText(im, notification, Point(1, 3 * size.height + 1), FONT_HERSHEY_COMPLEX, 0.6f,
			Scalar::all(255), 1, CV_AA);
	}
    draw_string(im,"d - redetection");
    tracker.timer.display_fps(im,Point(1,im.rows-1));
    imshow("face tracker",im);
    int c = waitKey(10);
    if(c == 'q')break;
    else if(c == 'd')tracker.reset();
  }
  destroyWindow("face tracker"); cam.release(); return 0;
}
//==============================================================================
