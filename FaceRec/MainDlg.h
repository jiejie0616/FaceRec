#pragma once

#include "opencv_hotshots/ft/ft.h"
#include "opencv_hotshots/ft/CvvImage.h"
#include <ml.h>  
#include "afxwin.h"

// MainDlg 对话框
class MainDlg : public CDialogEx
{
	DECLARE_DYNAMIC(MainDlg)

public:
	MainDlg(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~MainDlg();

// 对话框数据
	enum { IDD = IDD_MAIN_DIALOG };

// 工具函数
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
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	CFont txtFont;			// 字体信息
	face_tracker tmodel;	//face tracking class
	CvSVM emodel;			// emotion model
	VideoCapture videoCapture; // 摄像头

	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnopencamera();			// 点击开启摄像头
	afx_msg void OnTimer(UINT_PTR nIDEvent);			// 定时更新图片显示
	virtual BOOL OnInitDialog();						// 初始化
	afx_msg void OnBnClickedBtnaddperson();				// 点击增加人物按钮
	afx_msg void OnBnClickedBtndeleteperson();			// 点击删除所有人物按钮
	afx_msg void OnBnClickedBtnofacerec();				// 点击人脸识别按钮
	afx_msg void OnBnClickedBtnemotionrec();			// 点击表情识别按钮
	afx_msg void OnBnClickedBtnretrace();				// 点击重新跟踪按钮
	virtual void OnFinalRelease();						// 关闭窗口时释放资源
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);	// 修改字体颜色
	void ReleaseModel();				// 释放加载的模型
	afx_msg void OnBnClickedBtnswitchtrack();
};
