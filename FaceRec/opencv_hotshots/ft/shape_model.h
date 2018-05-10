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
  Mat p;             // չʾģ��ʱ����ԭʼ����ͶӰ��չʾ�ռ����ͶӰ����
  Mat V;             // ��������ģ�͵����Ͼ��󣬵�һ�ж�Ӧ�߶ȱ任�������к͵����зֱ��� x �� y ��ƽ��
  Mat e;             // �������Ͽռ��ڱ���ģ������ı�׼�����
  Mat C;             // ����֮ǰ��ע�����ӹ�ϵ����

  int npts(){return V.rows/2;}             //number of points in shape model

  void			// ���㼯ͶӰ��ò�����͵Ŀռ��У��Ա�ͶӰ��ÿ������ѡ��ĸ�������������Ȩ��
  calc_params(const vector<Point2f> &pts,  //points to compute parameters from
          const Mat weight = Mat(),    //weight of each point (nx1) CV_32F
          const float c_factor = 3.0); //clamping factor
          

  vector<Point2f>    // ͨ��������������ģ�͵Ĳ������� p��ͨ��V��e���룩�����ɵ㼯
  calc_shape();

  void set_identity_params();              //set @p to identity 

  Mat                                      //scaled rotation mat (2x2) CV_32F 
  rot_scale_align(const Mat &src,          //source points
          const Mat &dst);         //destination points

  Mat                                      //centered shape
  center_shape(const Mat &pts);            //shape to center

  void	// ���������ݼ���ѧϰ����ģ��
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

  Mat                                      //[x1;y1;...;xn;yn] (2nxN) n��ʾ������������N��ʾͼ��֡�� 
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
