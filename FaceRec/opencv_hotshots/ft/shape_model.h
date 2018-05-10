/*
  shape_model: A combined local-global 2D point distribution model
*/
#ifndef _FT_SHAPE_MODEL_HPP_
#define _FT_SHAPE_MODEL_HPP_
#include <opencv2/core/core.hpp>
#include <vector>
using namespace cv;
using namespace std;
//==============================================================================
class shape_model{                         //2d linear shape model
public:
  Mat p;             // 展示模型时，将原始坐标投影到展示空间里的投影矩阵
  Mat V;             // 描述人脸模型的联合矩阵，第一列对应尺度变换，第三列和第四列分别是 x 和 y 的平移
  Mat e;             // 存在联合空间内表情模型坐标的标准差矩阵
  Mat C;             // 描述之前标注的连接关系矩阵

  int npts(){return V.rows/2;}             //number of points in shape model

  void			// 将点集投影到貌似脸型的空间中，对被投影的每个点有选择的给出单独的置信权重
  calc_params(const vector<Point2f> &pts,  //points to compute parameters from
          const Mat weight = Mat(),    //weight of each point (nx1) CV_32F
          const float c_factor = 3.0); //clamping factor
          

  vector<Point2f>    // 通过解码用在人脸模型的参数向量 p（通过V和e编码）来生成点集
  calc_shape();

  void set_identity_params();              //set @p to identity 

  Mat                                      //scaled rotation mat (2x2) CV_32F 
  rot_scale_align(const Mat &src,          //source points
          const Mat &dst);         //destination points

  Mat                                      //centered shape
  center_shape(const Mat &pts);            //shape to center

  void	// 从脸型数据集中学习编码模型
  train(const vector<vector<Point2f> > &p, //N-example shapes
    const vector<Vec2i> &con = vector<Vec2i>(),//point-connectivity
    const float frac = 0.95,           //fraction of variation to retain
    const int kmax = 10);              //maximum number of modes to retain

  void 
  write(FileStorage &fs) const;            //file storage object to write to

  void 
  read(const FileNode& node);              //file storage node to read from

protected:
  void clamp(const float c = 3.0);         //clamping factor (or standard dev)

  Mat                                      //[x1;y1;...;xn;yn] (2nxN) n表示样本点数量，N表示图像帧数 
  pts2mat(const vector<vector<Point2f> > &p); //points to vectorise

  Mat                                      //procrustes aligned shapes/column
  procrustes(const Mat &X,                 //shapes to align
         const int itol = 100,         //maximum number of iterations
         const float ftol = 1e-6);     //convergence tolerance

  Mat                                      //rigid basis (2nx4) CV_32F
  calc_rigid_basis(const Mat &X);          //procrustes algned shapes/column
};
//==============================================================================
#endif
