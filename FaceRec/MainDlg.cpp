// MainDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "FaceRec.h"
#include "MainDlg.h"
#include "afxdialogex.h"


// MainDlg 对话框

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
const char* MODE_NAMES[] = { "初始状态", "人脸检测", "收集人脸", "训练样本", "人脸识别", "表情识别", "删除所有", "错误!" };
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

vector<Point2f> cPoints;      //an array that stores characteristic points
float featureScaler;       //used to calculate local feature
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

Ptr<FaceRecognizer> model;
vector<Mat> preprocessedFaces;
vector<int> faceLabels;
vector<string> faceNames;
Mat old_prepreprocessedFace;
double old_time = 0;
bool isShow = true;
bool isTrack = true;

//create tracker parameters
face_tracker_params p;



IMPLEMENT_DYNAMIC(MainDlg, CDialogEx)

MainDlg::MainDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(MainDlg::IDD, pParent)
{
	
}

MainDlg::~MainDlg()
{
}

void MainDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(MainDlg, CDialogEx)
	ON_BN_CLICKED(IDC_BTNOPENCAMERA, &MainDlg::OnBnClickedBtnopencamera)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_BTNADDPERSON, &MainDlg::OnBnClickedBtnaddperson)
	ON_BN_CLICKED(IDC_BTNDELETEPERSON, &MainDlg::OnBnClickedBtndeleteperson)
	ON_BN_CLICKED(IDC_BTNOFACEREC, &MainDlg::OnBnClickedBtnofacerec)
	ON_BN_CLICKED(IDC_BTNEMOTIONREC, &MainDlg::OnBnClickedBtnemotionrec)
	ON_BN_CLICKED(IDC_BTNRETRACE, &MainDlg::OnBnClickedBtnretrace)
	ON_WM_CTLCOLOR()
	ON_BN_CLICKED(IDC_BTNSWITCHTRACK, &MainDlg::OnBnClickedBtnswitchtrack)
END_MESSAGE_MAP()

// 工具函数
float MainDlg::getDist(cv::Point p1, cv::Point p2)
{
	float r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	return r;
}

float MainDlg::getDistX(cv::Point p1, cv::Point p2)
{
	float r = abs(p1.x - p2.x);
	return r;
}

float MainDlg::getDistY(cv::Point p1, cv::Point p2)
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
void MainDlg::initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
	if (videoCapture.isOpened())
		return;
	// 打开摄像头
	try {
		videoCapture.open(cameraNumber);
	}
	catch (cv::Exception &e) {}
	if (!videoCapture.isOpened()) {
		cerr << "ERROR: Could not access the camera!" << endl;
		exit(1);
	}

	// 设置摄像头宽高
	videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
	videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);
}

// 将图片画到控件
void MainDlg::DrawPicToHDC(IplImage *img, UINT ID)
{
	CDC *pDC = GetDlgItem(ID)->GetDC();
	HDC hDC = pDC->GetSafeHdc();
	CRect rect;
	GetDlgItem(ID)->GetClientRect(&rect);
	CvvImage cimg;
	cimg.CopyOf(img); // 复制图片
	cimg.DrawToHDC(hDC, &rect); // 将图片绘制到显示控件的指定区域内
	ReleaseDC(pDC);
}

// 画字符串
Rect MainDlg::drawString(Mat img, string text, Point coord, Scalar color, float fontScale , int thickness, int fontFace)
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

// MainDlg 消息处理程序
// 点击打开摄像头
void MainDlg::OnBnClickedBtnopencamera()
{
	// 打开摄像头
	int cameraNumber = 0;
	initWebcam(videoCapture, cameraNumber);

	// 加载表情模型
	if (emodel.get_support_vector_count() == 0) {
		string emotion_file = "model/emotion_model.yaml";
		emodel.load(emotion_file.c_str());
	}

	// 加载人脸模型
	if (tmodel.smodel.V.empty()) {
		string tracker_file = "model/tracker_model.yaml";
		tmodel = load_ft<face_tracker>(tracker_file.c_str());
	}

	m_mode = MODE_DETECTION;

	// 设置人脸跟踪参数
	p.robust = false;
	p.ssize.resize(3);
	p.ssize[0] = Size(21, 21);
	p.ssize[1] = Size(11, 11);
	p.ssize[2] = Size(5, 5);

	SetTimer(1, 25, NULL);		// 设置帧率
}

// 响应函数，更新图片
void MainDlg::OnTimer(UINT_PTR nIDEvent)
{
	if (!isShow)
		return;
	// 抓取摄像机帧
	Mat cameraFrame;
	videoCapture >> cameraFrame;
	if (cameraFrame.empty()) {
		// 显示提示信息
		char show[200];
		sprintf(show, "无法抓取摄像机");
		isShow = false;
		MessageBox((CString)show, _T("提示"), MB_OKCANCEL);
		isShow = true;
		exit(1);
	}

	// 拷贝
	Mat displayedFrame;
	cameraFrame.copyTo(displayedFrame);

	
	int identity = -1;			// 识别身份
	double similarity;			// 识别相似度
	string outputStr;			// 相似度

	
	Rect faceRect;								// 人脸区域
	Rect searchedLeftEye, searchedRightEye;		// 眼睛区域
	Point leftEye, rightEye;					// 眼睛位置
	Mat preprocessedFace;						// 处理后的脸

	// 人脸跟踪
	if (tmodel.track(cameraFrame, &faceRect, &leftEye, &rightEye, p)) {
		if (isTrack) {
			tmodel.draw(displayedFrame);
		}
		else {
			rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

			// Draw light-blue anti-aliased circles for the 2 eyes.
			Scalar eyeColor = CV_RGB(0, 255, 255);
			circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
			circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
		}
		
		// 对输入图像进行处理
		preprocessedFace = getPreprocessedFace(cameraFrame, faceWidth, preprocessLeftAndRightSeparately, faceRect, leftEye, rightEye);
	}

	// 是否检测到人脸
	bool gotFaceAndEyes = false;
	if (preprocessedFace.data)
		gotFaceAndEyes = true;

	if (m_mode == MODE_DETECTION) {
		// 不做任何事情
	}
	else if (m_mode == MODE_COLLECT_FACES) {
		if (gotFaceAndEyes) {
			// 前后两张图片差距
			double imageDiff = 10000000000.0;
			if (old_prepreprocessedFace.data) {
				imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
			}

			// 计算时间差
			double current_time = (double)getTickCount();
			double timeDiff_seconds = (current_time - old_time) / getTickFrequency();

			// 图片差距和时间间隔阈值
			if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
				// 同时存储镜面
				Mat mirroredFace;
				flip(preprocessedFace, mirroredFace, 1);

				// 添加到训练样本
				preprocessedFaces.push_back(preprocessedFace);
				preprocessedFaces.push_back(mirroredFace);
				faceLabels.push_back(m_selectedPerson);
				faceLabels.push_back(m_selectedPerson);

				// 保存脸数
				m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  

				// 显示脸部
				Mat displayedFaceRegion = displayedFrame(faceRect);
				displayedFaceRegion += CV_RGB(90, 90, 90);

				// 更新
				old_prepreprocessedFace = preprocessedFace;
				old_time = current_time;
			}
		}
	}
	else if (m_mode == MODE_TRAINING) {
		// 至少需要两个人
		bool haveEnoughData = true;
		if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
			if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0)) {
				haveEnoughData = false;
			}
		}
		if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
			// 显示提示信息
			char show[200];
			sprintf(show, "请收集更多的样本");
			isShow = false;
			MessageBox((CString)show, _T("警告"), MB_OKCANCEL);
			isShow = true;
			haveEnoughData = false;
		}

		if (haveEnoughData) {
			// 开始训练
			model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
			m_mode = MODE_FACEREC;		// 开始识别
		}
		else {
			// 还需要继续收集
			m_mode = MODE_COLLECT_FACES;
		}

	}
	else if (m_mode == MODE_FACEREC) {
		if (gotFaceAndEyes && (preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())) {

			// Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
			Mat reconstructedFace;
			reconstructedFace = reconstructFace(model, preprocessedFace);

			// Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
			similarity = getSimilarity(preprocessedFace, reconstructedFace);

			// 相似度阈值
			if (similarity < UNKNOWN_PERSON_THRESHOLD) {
				identity = model->predict(preprocessedFace);
				outputStr = faceNames[identity];
				Point drawPoint = faceRect.tl();
				drawPoint.x -= 100;
				drawString(displayedFrame, outputStr, drawPoint, CV_RGB(255, 0, 0));
			}
			else {
				// 无法确定
				outputStr = "识别失败";
			}

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
		faceNames.clear();

		old_prepreprocessedFace = Mat();

		// Restart in Detection mode.
		m_mode = MODE_DETECTION;
	}
	else if (m_mode == MODE_EMOTIONREC) {
		// 表情识别
		cPoints = tmodel.points;
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
		class_nr_int = (int)emodel.predict(DataMat);
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
		Point drawPoint = faceRect.tl();
		drawPoint.x -= 100;
		drawString(displayedFrame, currentEmotion, drawPoint, CV_RGB(255, 0, 0));
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
		help = "当你准备好之后就点击添加人物开始吧";
	else if (m_mode == MODE_COLLECT_FACES)
		help = "点击人脸识别对 " + toString(m_numPersons) + " 个人的 " + toString(preprocessedFaces.size() / 2) + " 个样本进行训练";
	else if (m_mode == MODE_TRAINING)
		help = "正在训练中...";
	else if (m_mode == MODE_FACEREC) {
		if (outputStr != "识别失败")
			help = "与 " + outputStr + " 的相似度为 " + toString(similarity+0.2);
		else
			help = "没有找到匹配的人";
	}
	else if (m_mode == MODE_EMOTIONREC) {
		help = "当前表情是 " + currentEmotion + ".";
	}
	if (help.length() > 0) {
		// 输出帮助文字
		CString tempStr(help.c_str());
		GetDlgItem(IDC_TXTHELP)->SetWindowTextW(tempStr);

	}

	// 显示当前模式
	if (m_mode >= 0 && m_mode < MODE_END) {
		string modeStr = "模式：" + string(MODE_NAMES[m_mode]);
		CString tempStr(modeStr.c_str());
		GetDlgItem(IDC_TXTMODE)->SetWindowTextW(tempStr);
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

	// 显示处理后的图片
	IplImage m_lplImage = (IplImage)displayedFrame;
	DrawPicToHDC(&m_lplImage, IDC_PIC_FACE);

	CDialogEx::OnTimer(nIDEvent);
}

// 初始化对话框
BOOL MainDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// TODO:  在此添加额外的初始化
	CWnd *pWnd;
	pWnd = GetDlgItem(IDC_PIC_FACE); //获取控件指针，IDC_BUTTON1为控件ID号
	// 设置图片大小，与摄像头一致
	pWnd->SetWindowPos(NULL, 50, 80, DESIRED_CAMERA_WIDTH, DESIRED_CAMERA_HEIGHT, SWP_NOZORDER | SWP_NOMOVE);

	// 设置字体
	txtFont.CreatePointFont(130, _T("宋体"));
	GetDlgItem(IDC_TXTMODE)->SetFont(&txtFont);
	GetDlgItem(IDC_TXTHELP)->SetFont(&txtFont);

	isTrack = true;

	return TRUE;  // return TRUE unless you set the focus to a control
	// 异常:  OCX 属性页应返回 FALSE
}

// 添加人物样本
void MainDlg::OnBnClickedBtnaddperson()
{
	string name;
	CString str;
	GetDlgItem(IDC_EDITNAME)->GetWindowText(str);
	name = CStringA(str);
	GetDlgItem(IDC_EDITNAME)->SetWindowText(_T(""));
	if (name == "") {
		isShow = false;
		AfxMessageBox(_T("请输入名字！"));
		isShow = true;
		return;
	}
	else {
		for (int i = 0; i < faceNames.size(); ++i) 
		{
			if (faceNames[i] == name) 
			{
				isShow = false;
				AfxMessageBox(_T("名字已存在"));
				isShow = true;
				return;
			}
		}
		// 若最后一个人没有添加样本，直接用这个
		if ((m_numPersons == 0) || (m_latestFaces[m_numPersons - 1] >= 0)) {
			// Add a new person.
			m_numPersons++;
			m_latestFaces.push_back(-1); // Allocate space for an extra person.
			faceNames.push_back(name);
		}
		// 更新当前选择的人
		m_selectedPerson = m_numPersons - 1;
		
		m_mode = MODE_COLLECT_FACES;
		isTrack = false;
	}
}

// 删除所有人物
void MainDlg::OnBnClickedBtndeleteperson()
{
	m_mode = MODE_DELETE_ALL;
	isTrack = true;
}

// 人脸识别
void MainDlg::OnBnClickedBtnofacerec()
{
	if (m_mode == MODE_COLLECT_FACES) {
		m_mode = MODE_TRAINING;
		isTrack = false;
	}
	else {
		isShow = false;
		AfxMessageBox(_T("请先收集样本！"));
		isShow = true;
	}
}

// 表情识别
void MainDlg::OnBnClickedBtnemotionrec()
{
	m_mode = MODE_EMOTIONREC;
	isTrack = false;
}

// 重新跟踪
void MainDlg::OnBnClickedBtnretrace()
{
	tmodel.reset();
	m_mode = MODE_DETECTION;
	isTrack = true;
}


void MainDlg::OnFinalRelease()
{
	// TODO:  在此添加专用代码和/或调用基类
	videoCapture.release();

	CDialogEx::OnFinalRelease();
}


HBRUSH MainDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);

	// TODO:  在此更改 DC 的任何特性
	if (pWnd->GetDlgCtrlID() == IDC_TXTMODE)
	{
		pDC->SetTextColor(RGB(0, 255, 0));
	}
	// TODO:  如果默认的不是所需画笔，则返回另一个画笔
	return hbr;
}


// 点击切换跟踪形态按钮
void MainDlg::OnBnClickedBtnswitchtrack()
{
	isTrack = !isTrack;
}
