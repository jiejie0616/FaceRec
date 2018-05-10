/*
	ft.hpp
	用于加载、保存对象数据
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
// 为了让保存和加载采用了序列化的用户自定义类变得容易，采用模块化函数定义了load_ft,save_ft函数
template <class T> 
T load_ft(const char* fname){
  T x; FileStorage f(fname,FileStorage::READ);
  f["ft object"] >> x; f.release(); return x;	// 定义与对象关联的标签都为 ft object
}
//==============================================================================
template<class T>
void save_ft(const char* fname,const T& x){
  FileStorage f(fname,FileStorage::WRITE);
  f << "ft object" << x; f.release();
}
//==============================================================================
// 为了使 FileStorage 类的序列化能正常工作，还需要定义write, read函数
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
