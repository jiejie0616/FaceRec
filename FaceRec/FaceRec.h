
// FaceRec.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CFaceRecApp: 
// �йش����ʵ�֣������ FaceRec.cpp
//

class CFaceRecApp : public CWinApp
{
public:
	CFaceRecApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CFaceRecApp theApp;