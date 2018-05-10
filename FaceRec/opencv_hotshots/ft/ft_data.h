/*
  ���������������ݶ�д
*/
#ifndef _FT_FT_DATA_HPP_
#define _FT_FT_DATA_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
using namespace cv;
using namespace std;
//==============================================================================
class ft_data{                             //face tracker data
public:
  vector<int> symmetry;                    // �����������������ά�����û��������������һ��
  vector<Vec2i> connections;               // ����һ����ͨ���沿����
  vector<string> imnames;                  // �洢ÿ��ͼ���ļ���
  vector<vector<Point2f> > points;         // �洢�������λ��
  
  inline int n_images(){return imnames.size();}

  Mat                                      //idx'th image
  get_image(const int idx,                 //index of image to get
        const int flag = 2);           //0=gray,1=gray+flip,2=rgb,3=rgb+flip

  vector<Point2f>                          //points for idx'th image
  get_points(const int idx,                //index of image
         const bool flipped = false);  //flip points?

  void                                     //removes samples which have missing
  rm_incomplete_samples();                 //point annotations
  
  void
  rm_sample(const int idx);                //remove idx'th sample

  void
  draw_points(Mat &im,                     //image to draw on
          const int idx,               //index of shape 
          const bool flipped = false,  //flip points?
          const Scalar color=CV_RGB(255,0,0),//color to draw points in
          const vector<int> &pts=vector<int>());//indices of points to draw

  void
  draw_sym(Mat &im,                        //image to draw on
       const int idx,                  //index of shape 
       const bool flipped = false,     //flip points?
       const vector<int> &pts=vector<int>());//indices of points to draw

  void
  draw_connect(Mat &im,                    //image to draw on
           const int idx,              //index of shape 
           const bool flipped = false, //flip points?
           const Scalar color=CV_RGB(0,0,255),//color to draw points in
           const vector<int> &con=vector<int>());//indices of connections
  
  void 
  write(FileStorage &fs) const;            //file storage object to write to

  void 
  read(const FileNode& node);              //file storage node to read from
};
//==============================================================================
#endif
