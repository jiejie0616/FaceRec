
// FaceRecDlg.cpp : ʵ���ļ�
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


// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
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


// CFaceRecDlg �Ի���


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


// CFaceRecDlg ��Ϣ�������

BOOL CFaceRecDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
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

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO:  �ڴ���Ӷ���ĳ�ʼ������
	

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
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

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CFaceRecDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CFaceRecDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

//=========================���ߺ���=============================================
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

// ��� CK+ ���ݼ���ť
void CFaceRecDlg::OnBnClickedBtndatabase()
{
	if (data.imnames.size() == 0) {
		// ����CK+��ע���� 
		string annotation_file = "model/annotations.yaml";
		data = load_ft<ft_data>(annotation_file.c_str());
		if (data.imnames.size() == 0){
			AfxMessageBox(_T("Data file does not contain any annotations."));
			return;
		}
		data.rm_incomplete_samples();
	}

	// ��ʾ��ʾ��Ϣ
	char show[200];
	sprintf(show, "ͼƬ��: %d\n��������: %d\n������: %d\n\n����˵����\nq   �˳�\np   ��һ��\no   ��һ��\nf   ��ת ", 
		data.imnames.size(), data.symmetry.size(), data.connections.size());
	int result = MessageBox((CString)show, _T("���ݼ����ӻ�"), MB_OKCANCEL);
	if (result != IDOK)
		return;
	// ���ӻ���ע����
	namedWindow("Annotations");
	int index = 0; bool flipped = false;
	while (1){
		Mat image;
		if (flipped)image = data.get_image(index, 3);
		else image = data.get_image(index, 2);			// ����ͼƬ
		data.draw_connect(image, index, flipped);			// ��ͨ
		data.draw_sym(image, index, flipped);				// �Գ�
		imshow("Annotations", image);
		int c = waitKey(0);			// q �˳���p ��һ�ţ�o ��һ��,f ��ת
		if (c == 'q')break;
		else if (c == 'p')index++;
		else if (c == 'o')index--;
		else if (c == 'f')flipped = !flipped;
		if (index < 0)index = 0;
		else if (index >= int(data.imnames.size()))index = data.imnames.size() - 1;
	}
	destroyWindow("Annotations");
}

// �����״ģ�Ͱ�ť
void CFaceRecDlg::OnBnClickedBtnshapemodel()
{
	// ��ʾ��ʾ��Ϣ
	char show[200];
	sprintf(show, "����˵����\nq   �˳�     ");
	int result = MessageBox((CString)show, _T("��״ģ�Ϳ��ӻ�"), MB_OKCANCEL);
	if (result != IDOK)
		return;

	if (smodel.V.empty()) {
		// ������״ģ��
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
		// ���ݷǸ��Ա仯��չʾ���������������� V.cols-3 ��
		for (int k = 4; k < smodel.V.cols; k++){
			for (int j = 0; j < int(val.size()); j++){
				Mat p = Mat::zeros(smodel.V.cols, 1, CV_32F);
				// ��������Ϊ�̶�ֵ���Ա���ͼ����������
				p.at<float>(0) = scale; p.at<float>(2) = tranx; p.at<float>(3) = trany;
				// ��ԭ����
				p.at<float>(k) = scale*val[j] * 3.0*sqrt(smodel.e.at<float>(k));
				p.copyTo(smodel.p); img = Scalar::all(255);
				char str[256]; sprintf(str, "mode: %d, val: %f sd", k - 3, val[j] / 3.0);
				draw_string(img, str);
				// ���ݽṹ�� p �е���Ϣ����ԭͼ������
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

// ����ſ�ģ�Ͱ�ť
void CFaceRecDlg::OnBnClickedBtnpatchmodel()
{
	// ��ʾ��ʾ��Ϣ
	char show[200];
	sprintf(show, "����˵����\nq   �˳�\np   ��һ�ſ�\no   ��һ�ſ�");
	int result = MessageBox((CString)show, _T("��״ģ�Ϳ��ӻ�"), MB_OKCANCEL);
	if (result != IDOK)
		return;

	if (pmodel.reference.empty()) {
		// �����ſ�ģ��
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
	// �����п�ģ�͸��ݱ�ע��λ������ͬһͼ����
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
		int c = waitKey(0);		// q �˳���p ��һ����ģ�ͣ�o ��һ����ģ��
		if (c == 'q')break;
		else if (c == 'p')i++;
		else if (c == 'o')i--;
		if (i < 0)i = 0; else if (i >= pmodel.n_patches())i = pmodel.n_patches() - 1;
	}
	destroyWindow("patch model");
}

// �������ʶ�����
void CFaceRecDlg::OnBnClickedBtnfacerec()
{
	MainDlg dlg;
	dlg.DoModal();
}
