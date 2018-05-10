
// FaceRecDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "FaceRec.h"
#include "FaceRecDlg.h"
#include "afxdialogex.h"

#include "opencv_hotshots/ft/ft.h"
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
using namespace std;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CFaceRecDlg 对话框


CFaceRecDlg::CFaceRecDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CFaceRecDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CFaceRecDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CFaceRecDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTNDATABASE, &CFaceRecDlg::OnBnClickedBtndatabase)
	ON_BN_CLICKED(IDC_BTNSHAPEMODEL, &CFaceRecDlg::OnBnClickedBtnshapemodel)
	ON_BN_CLICKED(IDC_BTNPATCHMODEL, &CFaceRecDlg::OnBnClickedBtnpatchmodel)
	ON_BN_CLICKED(IDC_BTNFACEREC, &CFaceRecDlg::OnBnClickedBtnfacerec)
END_MESSAGE_MAP()


// CFaceRecDlg 消息处理程序

BOOL CFaceRecDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO:  在此添加额外的初始化代码
	

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CFaceRecDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CFaceRecDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CFaceRecDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

//=========================工具函数=============================================
void
CFaceRecDlg::draw_string(Mat img,                       //image to draw on
const string text)             //text to draw
{
	Size size = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.6f, 1, NULL);
	putText(img, text, Point(0, size.height), FONT_HERSHEY_COMPLEX, 0.6f,
		Scalar::all(0), 1, CV_AA);
	putText(img, text, Point(1, size.height + 1), FONT_HERSHEY_COMPLEX, 0.6f,
		Scalar::all(255), 1, CV_AA);
}

Rect CFaceRecDlg::drawString(Mat img, string text, Point coord, Scalar color, float fontScale, int thickness, int fontFace)
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

void
CFaceRecDlg::draw_shape(Mat &img,
const vector<Point2f> &q,
const Mat &C)
{
	int n = q.size();
	for (int i = 0; i < C.rows; i++)
		line(img, q[C.at<int>(i, 0)], q[C.at<int>(i, 1)], CV_RGB(255, 0, 0), 2);
	for (int i = 0; i < n; i++)
		circle(img, q[i], 1, CV_RGB(0, 0, 0), 2, CV_AA);
}

float                                      //scaling factor
CFaceRecDlg::calc_scale(const Mat &X,                   //scaling basis vector
const float width)              //width of desired shape
{
	int n = X.rows / 2; float xmin = X.at<float>(0), xmax = X.at<float>(0);
	for (int i = 0; i < n; i++){
		xmin = min(xmin, X.at<float>(2 * i));
		xmax = max(xmax, X.at<float>(2 * i));
	}return width / (xmax - xmin);
}

int
CFaceRecDlg::calc_height(const Mat &X,
const float scale)
{
	int n = X.rows / 2;
	float ymin = scale*X.at<float>(1), ymax = scale*X.at<float>(1);
	for (int i = 0; i < n; i++){
		ymin = min(ymin, scale*X.at<float>(2 * i + 1));
		ymax = max(ymax, scale*X.at<float>(2 * i + 1));
	}return int(ymax - ymin + 0.5);
}
//==============================================================================

// 点击 CK+ 数据集按钮
void CFaceRecDlg::OnBnClickedBtndatabase()
{
	if (data.imnames.size() == 0) {
		// 加载CK+标注数据 
		string annotation_file = "model/annotations.yaml";
		data = load_ft<ft_data>(annotation_file.c_str());
		if (data.imnames.size() == 0){
			AfxMessageBox(_T("Data file does not contain any annotations."));
			return;
		}
		data.rm_incomplete_samples();
	}

	// 显示提示信息
	char show[200];
	sprintf(show, "图片数: %d\n特征点数: %d\n连接数: %d\n\n按键说明：\nq   退出\np   下一张\no   上一张\nf   翻转 ", 
		data.imnames.size(), data.symmetry.size(), data.connections.size());
	int result = MessageBox((CString)show, _T("数据集可视化"), MB_OKCANCEL);
	if (result != IDOK)
		return;
	// 可视化标注数据
	namedWindow("Annotations");
	int index = 0; bool flipped = false;
	while (1){
		Mat image;
		if (flipped)image = data.get_image(index, 3);
		else image = data.get_image(index, 2);			// 背景图片
		data.draw_connect(image, index, flipped);			// 连通
		data.draw_sym(image, index, flipped);				// 对称
		imshow("Annotations", image);
		int c = waitKey(0);			// q 退出，p 下一张，o 上一张,f 翻转
		if (c == 'q')break;
		else if (c == 'p')index++;
		else if (c == 'o')index--;
		else if (c == 'f')flipped = !flipped;
		if (index < 0)index = 0;
		else if (index >= int(data.imnames.size()))index = data.imnames.size() - 1;
	}
	destroyWindow("Annotations");
}

// 点击形状模型按钮
void CFaceRecDlg::OnBnClickedBtnshapemodel()
{
	// 显示提示信息
	char show[200];
	sprintf(show, "按键说明：\nq   退出     ");
	int result = MessageBox((CString)show, _T("形状模型可视化"), MB_OKCANCEL);
	if (result != IDOK)
		return;

	if (smodel.V.empty()) {
		// 加载形状模型
		string shape_file = "model/shape_model.yaml";
		smodel = load_ft<shape_model>(shape_file.c_str());
	}

	//compute rigid parameters
	int n = smodel.V.rows / 2;
	float scale = calc_scale(smodel.V.col(0), 200);
	float tranx = n*150.0 / smodel.V.col(2).dot(Mat::ones(2 * n, 1, CV_32F));
	float trany = n*150.0 / smodel.V.col(3).dot(Mat::ones(2 * n, 1, CV_32F));

	//generate trajectory of parameters
	vector<float> val;
	for (int i = 0; i < 50; i++)val.push_back(float(i) / 50);
	for (int i = 0; i < 50; i++)val.push_back(float(50 - i) / 50);
	for (int i = 0; i < 50; i++)val.push_back(-float(i) / 50);
	for (int i = 0; i < 50; i++)val.push_back(-float(50 - i) / 50);

	//visualise
	Mat img(300, 300, CV_8UC3); namedWindow("shape model");
	while (1){
		// 根据非刚性变化，展示动画，动画数量共 V.cols-3 个
		for (int k = 4; k < smodel.V.cols; k++){
			for (int j = 0; j < int(val.size()); j++){
				Mat p = Mat::zeros(smodel.V.cols, 1, CV_32F);
				// 以下三项为固定值，以便让图像处于正中央
				p.at<float>(0) = scale; p.at<float>(2) = tranx; p.at<float>(3) = trany;
				// 还原脸型
				p.at<float>(k) = scale*val[j] * 3.0*sqrt(smodel.e.at<float>(k));
				p.copyTo(smodel.p); img = Scalar::all(255);
				char str[256]; sprintf(str, "mode: %d, val: %f sd", k - 3, val[j] / 3.0);
				draw_string(img, str);
				// 根据结构体 p 中的信息，还原图像坐标
				vector<Point2f> q = smodel.calc_shape();
				draw_shape(img, q, smodel.C);
				imshow("shape model", img);
				if (waitKey(10) == 'q') 
				{
					destroyWindow("shape model");
					return;
				}
			}
		}
	}
}

// 点击团块模型按钮
void CFaceRecDlg::OnBnClickedBtnpatchmodel()
{
	// 显示提示信息
	char show[200];
	sprintf(show, "按键说明：\nq   退出\np   下一团块\no   上一团块");
	int result = MessageBox((CString)show, _T("形状模型可视化"), MB_OKCANCEL);
	if (result != IDOK)
		return;

	if (pmodel.reference.empty()) {
		// 加载团块模型
		string patch_file = "model/patch_model.yaml";
		pmodel = load_ft<patch_models>(patch_file.c_str());
	}
	
	//compute scale factor
	int width = 200;
	float scale = calc_scale(pmodel.reference, width);
	int height = calc_height(pmodel.reference, scale);

	//compute image width
	int max_width = 0, max_height = 0;
	for (int i = 0; i < pmodel.n_patches(); i++){
		Size size = pmodel.patches[i].patch_size();
		max_width = max(max_width, int(scale*size.width));
		max_height = max(max_width, int(scale*size.height));
	}
	//create reference image
	// 将所有块模型根据标注点位置贴到同一图像上
	Size image_size(width + 4 * max_width, height + 4 * max_height);
	Mat image(image_size.height, image_size.width, CV_8UC3);
	image = Scalar::all(255);
	vector<Point> points(pmodel.n_patches());
	vector<Mat> P(pmodel.n_patches());
	for (int i = 0; i < pmodel.n_patches(); i++){
		Mat I1; normalize(pmodel.patches[i].P, I1, 0, 255, NORM_MINMAX);
		Mat I2; resize(I1, I2, Size(scale*I1.cols, scale*I1.rows));
		Mat I3; I2.convertTo(I3, CV_8U); cvtColor(I3, P[i], CV_GRAY2RGB);
		points[i] = Point(scale*pmodel.reference.fl(2 * i) +
			image_size.width / 2 - P[i].cols / 2,
			scale*pmodel.reference.fl(2 * i + 1) +
			image_size.height / 2 - P[i].rows / 2);
		Mat I = image(Rect(points[i].x, points[i].y, P[i].cols, P[i].rows));
		P[i].copyTo(I);
	}
	//animate
	namedWindow("patch model");
	int i = 0;
	while (1){
		Mat img = image.clone();
		Mat I = img(Rect(points[i].x, points[i].y, P[i].cols, P[i].rows));
		P[i].copyTo(I);
		rectangle(img, points[i], Point(points[i].x + P[i].cols, points[i].y + P[i].rows),
			CV_RGB(255, 0, 0), 2, CV_AA);
		char str[256]; sprintf(str, "patch %d", i); draw_string(img, str);
		imshow("patch model", img);
		int c = waitKey(0);		// q 退出，p 下一个块模型，o 上一个块模型
		if (c == 'q')break;
		else if (c == 'p')i++;
		else if (c == 'o')i--;
		if (i < 0)i = 0; else if (i >= pmodel.n_patches())i = pmodel.n_patches() - 1;
	}
	destroyWindow("patch model");
}

// 点击人脸识别界面
void CFaceRecDlg::OnBnClickedBtnfacerec()
{
	MainDlg dlg;
	dlg.DoModal();
}
