/*
  patch_model: Correlation patch experts
*/
#ifndef _FT_PATCH_MODEL_HPP_
#define _FT_PATCH_MODEL_HPP_
#include "opencv_hotshots/ft/ft_data.h"
#include <opencv2/core/core.hpp>
#include <vector>
using namespace cv;
using namespace std;
//==============================================================================
class patch_model{                         //correlation-based patch expert
public:
  Mat P;                                   //normalised patch
  
  Size                                     //size of patch model
  patch_size(){return P.size();}

  Mat                                      //response map (CV_32F)
  calc_response(const Mat &im,             //image to compute response from
        const bool sum2one=false); //normalize response to sum to one?
  void 
  train(const vector<Mat> &images,         //feature centered training images
    const Size psize,                  //desired patch size
    const float var = 1.0,             //variance of annotation error
    const float lambda = 1e-6,         //regularization weight
    const float mu_init = 1e-3,        //initial stoch-grad step size
    const int nsamples = 1000,         //number of stoch-grad samples
    const bool visi = false);          //visualize intermediate results?

  void 
  write(FileStorage &fs) const;            //file storage object to write to

  void 
  read(const FileNode& node);              //file storage node to read from

protected:  
  Mat                                      //single channel log-scale image
  convert_image(const Mat &im);            //gray or rgb unsigned char image
};
//==============================================================================
class patch_models{                        //collection of patch experts
public:
  Mat reference;                           // 指定尺度变换和旋转的点集
  vector<patch_model> patches;             //patch models

  inline int                               //number of patches
  n_patches(){return patches.size();}

  void 
  train(ft_data &data,                     // 存放手工标注的数据
    const vector<Point2f> &ref,        // 指定大小和旋转角度的人脸特征的参考点集
    const Size psize,                  // 团块模型的大小
    const Size ssize,                  // 搜索窗口的大小，即在样本图像上可以搜索团块模型的范围
    const bool mirror = true,          // use mirrored images?
    const float var = 1.0,             //variance of annotation error
    const float lambda = 1e-6,         //regularization weight
    const float mu_init = 1e-3,        //initial stoch-grad step size
    const int nsamples = 1000,         //number of stoch-grad samples
    const bool visi = true);           //visualize intermediate results?

  vector<Point2f>                          //locations of peaks/feature
  calc_peaks(const Mat &im,                //image to detect features in
         const vector<Point2f> &points,//initial estimate of shape
         const Size ssize=Size(21,21));//search window size
  void 
  write(FileStorage &fs) const;            //file storage object to write to

  void 
  read(const FileNode& node);              //file storage node to read from

protected:
  Mat                                      //inverted similarity transform
  inv_simil(const Mat &S);                 //similarity transform

  Mat                                      //similarity tranform referece->pts
  calc_simil(const Mat &pts);              //destination shape

  vector<Point2f>                          //similarity transformed shape
  apply_simil(const Mat &S,                //similarity transform
          const vector<Point2f> &points); //shape to transform
};
//==============================================================================
#endif
