/*
	ft.hpp
	���ڼ��ء������������
*/

#ifndef _FT_FT_HPP_
#define _FT_FT_HPP_
#include "opencv_hotshots/ft/ft_data.h"
#include "opencv_hotshots/ft/patch_model.h"
#include "opencv_hotshots/ft/shape_model.h"
#include "opencv_hotshots/ft/face_detector.h"
#include "opencv_hotshots/ft/face_tracker.h"
#include "opencv_hotshots/ft/preprocessFace.h"     // Easily preprocess face images, for face recognition.
#include "opencv_hotshots/ft/recognition.h"     // Train the face recognition system and recognize a person from an image.

#include "opencv_hotshots/ft/ImageUtils.h"      // Shervin's handy OpenCV utility functions.
//==============================================================================
// Ϊ���ñ���ͼ��ز��������л����û��Զ����������ף�����ģ�黯����������load_ft,save_ft����
template <class T> 
T load_ft(const char* fname){
  T x; FileStorage f(fname,FileStorage::READ);
  f["ft object"] >> x; f.release(); return x;	// �������������ı�ǩ��Ϊ ft object
}
//==============================================================================
template<class T>
void save_ft(const char* fname,const T& x){
  FileStorage f(fname,FileStorage::WRITE);
  f << "ft object" << x; f.release();
}
//==============================================================================
// Ϊ��ʹ FileStorage ������л�����������������Ҫ����write, read����
template<class T>
void 
write(FileStorage& fs, 
      const string&, 
      const T& x)
{
  x.write(fs);
}
//==============================================================================
template<class T>
void 
read(const FileNode& node, 
     T& x,
     const T& d)
{
  if(node.empty())x = d; else x.read(node);
}
//==============================================================================
#endif
