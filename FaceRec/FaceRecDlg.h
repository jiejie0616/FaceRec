
// FaceRecDlg.h : 头文件
//

#pragma once

#include "opencv_hotshots/ft/ft.h"
#include <ml.h>  
#include "MainDlg.h"

#define fl at<float>
#define VK_ESCAPE 0x1B 
#define VK_SPACE  0x20

// CFaceRecDlg 对话框
class CFaceRecDlg : public CDialogEx
{
// 构造
public:
	CFaceRecDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_FACEREC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持

// 工具函数
private:
	void
		draw_string(Mat img,                       //image to draw on
		const string text);             //text to draw

	Rect drawString(Mat img,
		string text,
		Point coord,
		Scalar color,
		float fontScale = 0.6f,
		int thickness = 1,
		int fontFace = FONT_HERSHEY_COMPLEX);

	void
		draw_shape(Mat &img,
		const vector<Point2f> &q,
		const Mat &C);

	float                                      //scaling factor
		calc_scale(const Mat &X,                   //scaling basis vector
		const float width);              //width of desired shape

	int
		calc_height(const Mat &X,
		const float scale);

// 实现
protected:
	HICON m_hIcon;

	ft_data data;			//face tracker data
	shape_model smodel;		//2d linear shape model
	patch_models pmodel;	//collection of patch experts

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtndatabase();		// 点击 CK+ 数据集
	afx_msg void OnBnClickedBtnshapemodel();	// 点击形状模型
	afx_msg void OnBnClickedBtnpatchmodel();	// 点击团块模型
	afx_msg void OnBnClickedBtnfacerec();		// 点击人脸识别
};
