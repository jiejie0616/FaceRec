
// FaceRecDlg.h : ͷ�ļ�
//

#pragma once

#include "opencv_hotshots/ft/ft.h"
#include <ml.h>  
#include "MainDlg.h"

#define fl at<float>
#define VK_ESCAPE 0x1B 
#define VK_SPACE  0x20

// CFaceRecDlg �Ի���
class CFaceRecDlg : public CDialogEx
{
// ����
public:
	CFaceRecDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_FACEREC_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��

// ���ߺ���
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

// ʵ��
protected:
	HICON m_hIcon;

	ft_data data;			//face tracker data
	shape_model smodel;		//2d linear shape model
	patch_models pmodel;	//collection of patch experts

	// ���ɵ���Ϣӳ�亯��
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtndatabase();		// ��� CK+ ���ݼ�
	afx_msg void OnBnClickedBtnshapemodel();	// �����״ģ��
	afx_msg void OnBnClickedBtnpatchmodel();	// ����ſ�ģ��
	afx_msg void OnBnClickedBtnfacerec();		// �������ʶ��
};
