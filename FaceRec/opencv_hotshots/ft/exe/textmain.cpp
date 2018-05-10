#include <stdio.h>
#include <ml.h>  
#include <vector>
#include <string>
#include <iostream>

// Include OpenCV's C++ Interface
#include "opencv2/opencv.hpp"

// Include the rest of our code!
#include "opencv_hotshots/ft/svm.h"
#include "opencv_hotshots/ft/ft.h"
#include <opencv2/highgui/highgui.hpp>
//#include "opencv_hotshots/ft/detectObject.h"       // Easily detect faces or eyes (using LBP or Haar Cascades).
#include "opencv_hotshots/ft/preprocessFace.h"     // Easily preprocess face images, for face recognition.
#include "opencv_hotshots/ft/recognition.h"     // Train the face recognition system and recognize a person from an image.

#include "opencv_hotshots/ft/ImageUtils.h"      // Shervin's handy OpenCV utility functions.

using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
#define VK_ESCAPE 0x1B      // Escape character (27)
#define VK_SPACE  0x20
#endif

// 人脸识别算法
const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";


// 人脸识别阀值，小于阀值才算识别成功
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;


// 人脸宽度，高度
const int faceWidth = 70;
const int faceHeight = faceWidth;

// 窗口大小
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// 图片收集参数
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // 前后两张图片相差多少可以采集
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       // 采集间隔

const char *windowName = "FaceRec_Linjie";   // 窗口名字
const int BORDER = 8;  // GUI 间隔

const bool preprocessLeftAndRightSeparately = true;


// Running mode for the Webcam-based interactive GUI program.
enum MODES { MODE_STARTUP = 0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_FACEREC, MODE_EMOTIONREC, MODE_DELETE_ALL, MODE_END };
const char* MODE_NAMES[] = { "Startup", "Detection", "Collect Faces", "Training", "FaceRec", "EmoRec", "Delete All", "ERROR!" };
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

// Position of GUI buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnFacRec;
Rect m_rcBtnEmoRec;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

vector<Point2f> cPoints;      //an array that stores characteristic points
float featureScaler;       //used to calculate local feature
CvSVM svm;
cv::Point textArea;
string currentEmotion;

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

// C++ conversion functions between integers (or floats) to std::string.
template <typename T> string toString(T t)
{
	ostringstream out;
	out << t;
	return out.str();
}

template <typename T> T fromString(string t)
{
	T out;
	istringstream in(t);
	in >> out;
	return out;
}

// 初始化摄像头
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
	// 打开摄像头
	try {
		videoCapture.open(cameraNumber);
	}
	catch (cv::Exception &e) {}
	if (!videoCapture.isOpened()) {
		cerr << "ERROR: Could not access the camera!" << endl;
		exit(1);
	}
	cout << "Loaded camera " << cameraNumber << "." << endl;
}


// 往图片上写文字
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
	// Get the text size & baseline.
	int baseline = 0;
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;

	// Adjust the coords for left/right-justified or top/bottom-justified.
	if (coord.y >= 0) {
		// Coordinates are for the top-left corner of the text from the top-left of the image, so move down by one row.
		coord.y += textSize.height;
	}
	else {
		// Coordinates are for the bottom-left corner of the text from the bottom-left of the image, so come up from the bottom.
		coord.y += img.rows - baseline + 1;
	}
	// Become right-justified if desired.
	if (coord.x < 0) {
		coord.x += img.cols - textSize.width + 1;
	}

	// Get the bounding box around the text.
	Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

	// Draw anti-aliased text.
	putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

	// Let the user know how big their text is, in case they want to arrange things.
	return boundingRect;
}

// 往图片上画按钮
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
	int B = BORDER;
	Point textCoord = Point(coord.x + B, coord.y + B);
	// Get the bounding box around the text.
	Rect rcText = drawString(img, text, textCoord, CV_RGB(0, 0, 0));
	// Draw a filled rectangle around the text.
	Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2 * B, rcText.height + 2 * B);
	// Set a minimum button width.
	if (rcButton.width < minWidth)
		rcButton.width = minWidth;
	// Make a semi-transparent white rectangle.
	Mat matButton = img(rcButton);
	matButton += CV_RGB(90, 90, 90);
	// Draw a non-transparent white border.
	rectangle(img, rcButton, CV_RGB(200, 200, 200), 1, CV_AA);

	// Draw the actual text that will be displayed, using anti-aliasing.
	drawString(img, text, textCoord, CV_RGB(10, 55, 20));

	return rcButton;
}

// 判断是否点到 UI 上了
bool isPointInRect(const Point pt, const Rect rc)
{
	if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
		if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
			return true;

	return false;
}

// 鼠标点击事件
void onMouse(int event, int x, int y, int, void*)
{
	// 只检测鼠标左键事件
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	// 检测是否点到 UI 上了
	Point pt = Point(x, y);
	if (isPointInRect(pt, m_rcBtnAdd)) {		// 点击 Add Person 按钮
		cout << "User clicked [Add Person] button when numPersons was " << m_numPersons << endl;
		// 若最后一个人没有添加样本，直接用这个
		if ((m_numPersons == 0) || (m_latestFaces[m_numPersons - 1] >= 0)) {
			// Add a new person.
			m_numPersons++;
			m_latestFaces.push_back(-1); // Allocate space for an extra person.
			cout << "Num Persons: " << m_numPersons << endl;
		}
		// 更新当前选择的人
		m_selectedPerson = m_numPersons - 1;
		m_mode = MODE_COLLECT_FACES;
	}
	else if (isPointInRect(pt, m_rcBtnDel)) {
		cout << "User clicked [Delete All] button." << endl;
		m_mode = MODE_DELETE_ALL;
	}
	else if (isPointInRect(pt, m_rcBtnFacRec)) {
		cout << "User clicked [FaceRec] button." << endl;
		if (m_mode == MODE_COLLECT_FACES) {
			cout << "User wants to begin training." << endl;
			m_mode = MODE_TRAINING;
		}
		else {
			cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ..." << endl;
		}
	}
	else if (isPointInRect(pt, m_rcBtnEmoRec)) {
		cout << "User clicked [EmoRec] button." << endl;
		m_mode = MODE_EMOTIONREC;
	}
	else {
		cout << "User clicked on the image" << endl;
		// Check if the user clicked on one of the faces in the list.
		int clickedPerson = -1;
		for (int i = 0; i < m_numPersons; i++) {
			if (m_gui_faces_top >= 0) {
				Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
				if (isPointInRect(pt, rcFace)) {
					clickedPerson = i;
					break;
				}
			}
		}
		// Change the selected person, if the user clicked on a face in the GUI.
		if (clickedPerson >= 0) {
			// Change the current person, and collect more photos for them.
			m_selectedPerson = clickedPerson; // Use the newly added person.
			m_mode = MODE_COLLECT_FACES;
		}
		// Otherwise they clicked in the center.
		else {
			// Change to training mode if it was collecting faces.
			//             if (m_mode == MODE_COLLECT_FACES) {
			//                 cout << "User wants to begin training." << endl;
			//                 m_mode = MODE_TRAINING;
			//             }
		}
	}
}


// Main loop that runs forever, until the user hits Escape to quit.
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture)
{
	Ptr<FaceRecognizer> model;
	vector<Mat> preprocessedFaces;
	vector<int> faceLabels;
	Mat old_prepreprocessedFace;
	double old_time = 0;

	// Since we have already initialized everything, lets start in Detection mode.
	m_mode = MODE_DETECTION;

	//load detector model
	face_tracker tracker = load_ft<face_tracker>("tracker_model2.yaml");

	//create tracker parameters
	face_tracker_params p; p.robust = false;
	p.ssize.resize(3);
	p.ssize[0] = Size(21, 21);
	p.ssize[1] = Size(11, 11);
	p.ssize[2] = Size(5, 5);

	// Run forever, until the user hits Escape to "break" out of this loop.
	while (true) {

		// Grab the next camera frame. Note that you can't modify camera frames.
		Mat cameraFrame;
		videoCapture >> cameraFrame;
		if (cameraFrame.empty()) {
			cerr << "ERROR: Couldn't grab the next camera frame." << endl;
			exit(1);
		}

		// Get a copy of the camera frame that we can draw onto.
		Mat displayedFrame;
		cameraFrame.copyTo(displayedFrame);

		// Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
		int identity = -1;

		// Find a face and preprocess it to have a standard size and contrast & brightness.
		Rect faceRect;  // Position of detected face.
		Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
		Point leftEye, rightEye;    // Position of the detected eyes.
		Mat preprocessedFace;

		if (tracker.track(cameraFrame, &faceRect, &leftEye, &rightEye, p)) {
			tracker.draw(displayedFrame);
			preprocessedFace = getPreprocessedFace(cameraFrame, faceWidth, preprocessLeftAndRightSeparately, faceRect, leftEye, rightEye);
		}

		// Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, preprocessLeftAndRightSeparately, faceRect, leftEye, rightEye);
		// Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

		bool gotFaceAndEyes = false;
		if (preprocessedFace.data)
			gotFaceAndEyes = true;

		// Draw an anti-aliased rectangle around the detected face.
		//         if (faceRect.width > 0) {
		//             rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
		// 
		//             // Draw light-blue anti-aliased circles for the 2 eyes.
		// //             Scalar eyeColor = CV_RGB(0,255,255);
		// //             if (leftEye.x >= 0) {   // Check if the eye was detected
		// //                 circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
		// //             }
		// //             if (rightEye.x >= 0) {   // Check if the eye was detected
		// //                 circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
		// //             }
		//         }

		if (m_mode == MODE_DETECTION) {
			// Don't do anything special.
		}
		else if (m_mode == MODE_COLLECT_FACES) {
			// Check if we have detected a face.
			if (gotFaceAndEyes) {


				// Check if this face looks somewhat different from the previously collected face.
				double imageDiff = 10000000000.0;
				if (old_prepreprocessedFace.data) {
					imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
				}

				// Also record when it happened.
				double current_time = (double)getTickCount();
				double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

				// Only process the face if it is noticeably different from the previous frame and there has been noticeable time gap.
				if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
					// Also add the mirror image to the training set, so we have more training data, as well as to deal with faces looking to the left or right.
					Mat mirroredFace;
					flip(preprocessedFace, mirroredFace, 1);

					// Add the face images to the list of detected faces.
					preprocessedFaces.push_back(preprocessedFace);
					preprocessedFaces.push_back(mirroredFace);
					faceLabels.push_back(m_selectedPerson);
					faceLabels.push_back(m_selectedPerson);

					// Keep a reference to the latest face of each person.
					m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  // Point to the non-mirrored face.
					// Show the number of collected faces. But since we also store mirrored faces, just show how many the user thinks they stored.
					cout << "Saved face " << (preprocessedFaces.size() / 2) << " for person " << m_selectedPerson << endl;

					// Make a white flash on the face, so the user knows a photo has been taken.
					Mat displayedFaceRegion = displayedFrame(faceRect);
					displayedFaceRegion += CV_RGB(90, 90, 90);

					// Keep a copy of the processed face, to compare on next iteration.
					old_prepreprocessedFace = preprocessedFace;
					old_time = current_time;
				}
			}
		}
		else if (m_mode == MODE_TRAINING) {

			// Check if there is enough data to train from. For Eigenfaces, we can learn just one person if we want, but for Fisherfaces,
			// we need atleast 2 people otherwise it will crash!
			bool haveEnoughData = true;
			if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
				if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0)) {
					cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ..." << endl;
					haveEnoughData = false;
				}
			}
			if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
				cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
				haveEnoughData = false;
			}

			if (haveEnoughData) {
				// Start training from the collected faces using Eigenfaces or a similar algorithm.
				model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);

				// Now that training is over, we can start recognizing!
				m_mode = MODE_FACEREC;
			}
			else {
				// Since there isn't enough training data, go back to the face collection mode!
				m_mode = MODE_COLLECT_FACES;
			}

		}
		else if (m_mode == MODE_FACEREC) {
			if (gotFaceAndEyes && (preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())) {

				// Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
				Mat reconstructedFace;
				reconstructedFace = reconstructFace(model, preprocessedFace);

				// Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
				double similarity = getSimilarity(preprocessedFace, reconstructedFace);

				string outputStr;
				if (similarity < UNKNOWN_PERSON_THRESHOLD) {
					// Identify who the person is in the preprocessed face image.
					identity = model->predict(preprocessedFace);
					outputStr = toString(identity);
				}
				else {
					// Since the confidence is low, assume it is an unknown person.
					outputStr = "Unknown";
				}
				cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;

				// Show the confidence rating for the recognition in the mid-top of the display.
				int cx = (displayedFrame.cols - faceWidth) / 2;
				Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
				Point ptTopLeft = Point(cx - 15, BORDER);
				// Draw a gray line showing the threshold for an "unknown" person.
				Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
				rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200, 200, 200), 1, CV_AA);
				// Crop the confidence rating between 0.0 to 1.0, to show in the bar.
				double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
				Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
				// Show the light-blue confidence bar.
				rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0, 255, 255), CV_FILLED, CV_AA);
				// Show the gray border of the bar.
				rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200, 200, 200), 1, CV_AA);
			}
		}
		else if (m_mode == MODE_DELETE_ALL) {
			// Restart everything!
			m_selectedPerson = -1;
			m_numPersons = 0;
			m_latestFaces.clear();
			preprocessedFaces.clear();
			faceLabels.clear();
			old_prepreprocessedFace = Mat();

			// Restart in Detection mode.
			m_mode = MODE_DETECTION;
		}
		else if (m_mode == MODE_EMOTIONREC) {
			// 表情识别
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
			node[9]= lipTightener;
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
			if (class_nr_int == 2) {
				currentEmotion = "Angry";
			}
			else if (class_nr_int == 1){
				currentEmotion = "Happy";
			}
			else if (class_nr_int == 3){
				currentEmotion = "Disgust";
			}
			else if (class_nr_int == -1){
				currentEmotion = "Sad";
			}
			else if (class_nr_int == -2){
				currentEmotion = "Surprise";
			}
			else if (class_nr_int == -3){
				currentEmotion = "Fear";
			}
			else
			{
				currentEmotion = "Neutral";
			}
		}
		else {
			cerr << "ERROR: Invalid run mode " << m_mode << endl;
			exit(1);
		}


		// Show the help, while also showing the number of collected faces. Since we also collect mirrored faces, we should just
		// tell the user how many faces they think we saved (ignoring the mirrored faces), hence divide by 2.
		string help;
		Rect rcHelp;
		if (m_mode == MODE_DETECTION)
			help = "Click [Add Person] when ready to collect faces.";
		else if (m_mode == MODE_COLLECT_FACES)
			help = "Click [FaceRec] to train from your " + toString(preprocessedFaces.size() / 2) + " faces of " + toString(m_numPersons) + " people.";
		else if (m_mode == MODE_TRAINING)
			help = "Please wait while your " + toString(preprocessedFaces.size() / 2) + " faces of " + toString(m_numPersons) + " people builds.";
		else if (m_mode == MODE_FACEREC)
			help = "Click people on the right to add more faces to them, or [Add Person] for someone new.";
		else if (m_mode == MODE_EMOTIONREC) {
			help = "The current emotion is " + currentEmotion + ".";
		}
		if (help.length() > 0) {
			// Draw it with a black background and then again with a white foreground.
			// Since BORDER may be 0 and we need a negative position, subtract 2 from the border so it is always negative.
			float txtSize = 0.4;
			drawString(displayedFrame, help, Point(BORDER, -BORDER - 2), CV_RGB(0, 0, 0), txtSize);  // Black shadow.
			rcHelp = drawString(displayedFrame, help, Point(BORDER + 1, -BORDER - 1), CV_RGB(255, 255, 255), txtSize);  // White text.
		}

		// Show the current mode.
		if (m_mode >= 0 && m_mode < MODE_END) {
			string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
			drawString(displayedFrame, modeStr, Point(BORDER, -BORDER - 2 - rcHelp.height), CV_RGB(0, 0, 0));       // Black shadow
			drawString(displayedFrame, modeStr, Point(BORDER + 1, -BORDER - 1 - rcHelp.height), CV_RGB(0, 255, 0)); // Green text
		}

		// Show the current preprocessed face in the top-center of the display.
		int cx = (displayedFrame.cols - faceWidth) / 2;
		if (preprocessedFace.data) {
			// Get a BGR version of the face, since the output is BGR color.
			Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
			cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
			// Get the destination ROI (and make sure it is within the image!).
			//min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
			Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
			Mat dstROI = displayedFrame(dstRC);
			// Copy the pixels from src to dst.
			srcBGR.copyTo(dstROI);
		}
		// Draw an anti-aliased border around the face, even if it is not shown.
		rectangle(displayedFrame, Rect(cx - 1, BORDER - 1, faceWidth + 2, faceHeight + 2), CV_RGB(200, 200, 200), 1, CV_AA);

		// Draw the GUI buttons into the main image.
		m_rcBtnAdd = drawButton(displayedFrame, "Add Person", Point(BORDER, BORDER));
		m_rcBtnDel = drawButton(displayedFrame, "Delete All", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
		m_rcBtnFacRec = drawButton(displayedFrame, "FaceRec", Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);
		m_rcBtnEmoRec = drawButton(displayedFrame, "EmoRec", Point(m_rcBtnFacRec.x, m_rcBtnFacRec.y + m_rcBtnFacRec.height), m_rcBtnAdd.width);

		// Show the most recent face for each of the collected people, on the right side of the display.
		m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
		m_gui_faces_top = BORDER;
		for (int i = 0; i < m_numPersons; i++) {
			int index = m_latestFaces[i];
			if (index >= 0 && index < (int)preprocessedFaces.size()) {
				Mat srcGray = preprocessedFaces[index];
				if (srcGray.data) {
					// Get a BGR version of the face, since the output is BGR color.
					Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
					cvtColor(srcGray, srcBGR, CV_GRAY2BGR);
					// Get the destination ROI (and make sure it is within the image!).
					int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
					Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
					Mat dstROI = displayedFrame(dstRC);
					// Copy the pixels from src to dst.
					srcBGR.copyTo(dstROI);
				}
			}
		}

		// Highlight the person being collected, using a red rectangle around their face.
		if (m_mode == MODE_COLLECT_FACES) {
			if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
				int y = min(m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows - faceHeight);
				Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
				rectangle(displayedFrame, rc, CV_RGB(255, 0, 0), 3, CV_AA);
			}
		}

		// Highlight the person that has been recognized, using a green rectangle around their face.
		if (identity >= 0 && identity < 1000) {
			int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
			Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
			rectangle(displayedFrame, rc, CV_RGB(0, 255, 0), 3, CV_AA);
		}

		// Show the camera frame on the screen.
		imshow(windowName, displayedFrame);


		// IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
		// Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
		char keypress = waitKey(20);  // This is needed if you want to see anything!

		if (keypress == VK_ESCAPE) {   // Escape Key
			// Quit the program!
			break;
		}
		else if (keypress == VK_SPACE) {
			tracker.reset();
		}

	}//end while
}


int main(int argc, char *argv[])
{
	VideoCapture videoCapture;

	cout << "WebcamFaceRec, by Linjie." << endl;
	cout << "Realtime face detection + face recognition from a webcam using LBP and Eigenfaces or Fisherfaces." << endl;
	cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

	cout << endl;
	cout << "Hit 'Escape' in the GUI window to quit." << endl;

	// Allow the user to specify a camera number, since not all computers will be the same camera number.
	int cameraNumber = 0;   // Change this if you want to use a different camera device.S
	// Get access to the webcam.
	initWebcam(videoCapture, cameraNumber);

	//load svm model
	svm.load("emotion.yaml");

	// Try to set the camera resolution. Note that this only works for some cameras on
	// some computers and only for some drivers, so don't rely on it to work!
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

	// Create a GUI window for display on the screen.
	namedWindow(windowName); // Resizable window, might not work on Windows.
	// Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
	setMouseCallback(windowName, onMouse, 0);

	// Run Face Recogintion interactively from the webcam. This function runs until the user quits.
	recognizeAndTrainUsingWebcam(videoCapture);

	return 0;
}
