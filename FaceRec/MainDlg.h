#pragma once

#include "opencv_hotshots/ft/ft.h"
#include "opencv_hotshots/ft/CvvImage.h"
#include <ml.h>  
#include "afxwin.h"

// MainDlg �Ի���
class MainDlg : public CDialogEx
{
	DECLARE_DYNAMIC(MainDlg)

public:
	MainDlg(CWnd* pParent = NULL);   // ��׼���캯��
	virtual ~MainDlg();

// �Ի�������
	enum { IDD = IDD_MAIN_DIALOG };

// ���ߺ���
private:
	float getDist(cv::Point p1, cv::Point p2);
	float getDistX(cv::Point p1, cv::Point p2);
	float getDistY(cv::Point p1, cv::Point p2);
	void initWebcam(VideoCapture &videoCapture, int cameraNumber);
	void DrawPicToHDC(IplImage *img, UINT ID);
	Rect drawString(Mat img,
		string text,
		Point coord,
		Scalar color,
		float fontScale = 0.6f,
		int thickness = 1,
		int fontFace = FONT_HERSHEY_COMPLEX);

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

	CFont txtFont;			// ������Ϣ
	face_tracker tmodel;	//face tracking class
	CvSVM emodel;			// emotion model
	VideoCapture videoCapture; // ����ͷ

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnopencamera();			// �����������ͷ
	afx_msg void OnTimer(UINT_PTR nIDEvent);			// ��ʱ����ͼƬ��ʾ
	virtual BOOL OnInitDialog();						// ��ʼ��
	afx_msg void OnBnClickedBtnaddperson();				// ����������ﰴť
	afx_msg void OnBnClickedBtndeleteperson();			// ���ɾ���������ﰴť
	afx_msg void OnBnClickedBtnofacerec();				// �������ʶ��ť
	afx_msg void OnBnClickedBtnemotionrec();			// �������ʶ��ť
	afx_msg void OnBnClickedBtnretrace();				// ������¸��ٰ�ť
	virtual void OnFinalRelease();						// �رմ���ʱ�ͷ���Դ
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);	// �޸�������ɫ
	void ReleaseModel();				// �ͷż��ص�ģ��
	afx_msg void OnBnClickedBtnswitchtrack();
};
