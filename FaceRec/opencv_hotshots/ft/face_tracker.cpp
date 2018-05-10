/*
  face_tracker: face tracking classes
*/
#include "opencv_hotshots/ft/face_tracker.h"
#include "opencv_hotshots/ft/ft.h"
#include <iostream>
#include "stdio.h"      // For 'sprintf()'
#define fl at<float>
//==============================================================================
//==============================================================================
//==============================================================================
//=============================== fps_timer ====================================
//==============================================================================
//==============================================================================
//==============================================================================
void 
fps_timer::
increment()
{
  if(fnum >= 29) {
    t_end = cv::getTickCount();
    fps = 30.0/(float(t_end-t_start)/getTickFrequency()); 
    t_start = t_end; fnum = 0;
  }else fnum += 1;
}
//==============================================================================
void 
fps_timer::
reset(){
t_start = cv::getTickCount(); fps = 0; fnum = 0;
}
//==============================================================================
void 
fps_timer::
display_fps(Mat &im,
        Point p)
{
  char str[256]; Point pt; if(p.y < 0)pt = Point(10,im.rows-20); else pt = p;
  sprintf(str,"%d frames/sec",(int)cvRound(fps)); string text = str;
  putText(im,text,pt,FONT_HERSHEY_SIMPLEX,0.5,Scalar::all(255));
}
//==============================================================================
//==============================================================================
//==============================================================================
//========================== face_tracker_params ===============================
//==============================================================================
//==============================================================================
//==============================================================================
face_tracker_params::
face_tracker_params()
{
  ssize.resize(3); 
  ssize[0] = Size(21,21); ssize[1] = Size(11,11); ssize[2] = Size(5,5);
  robust = false; itol = 20; ftol = 1e-3;
  scaleFactor = 1.1; minNeighbours = 2; minSize = Size(30,30);
}
//==============================================================================
void 
face_tracker_params::
write(FileStorage &fs) const
{
  assert(fs.isOpened()); fs << "{";
  fs << "nlevels" << int(ssize.size());
  for(int i = 0; i < int(ssize.size()); i++){ char str[256]; const char* ss;
    sprintf(str,"w %d",i); ss = str; fs << ss << ssize[i].width;
    sprintf(str,"h %d",i); ss = str; fs << ss << ssize[i].height;
  }
  fs << "robust" << robust
     << "itol" << itol
     << "ftol" << ftol
     << "scaleFactor" << scaleFactor
     << "minNeighbours" << minNeighbours
     << "minWidth" << minSize.width
     << "minHeight" << minSize.height
     << "}";
}
//==============================================================================
void 
face_tracker_params::
read(const FileNode& node)
{
  assert(node.type() == FileNode::MAP);
  int n; node["nlevels"] >> n; ssize.resize(n);
  for(int i = 0; i < n; i++){ char str[256]; const char* ss;
    sprintf(str,"w %d",i); ss = str; node[ss] >> ssize[i].width;
    sprintf(str,"h %d",i); ss = str; node[ss] >> ssize[i].height;
  }
  node["robust"] >> robust;
  node["itol"] >> itol;
  node["ftol"] >> ftol;
  node["scaleFactor"] >> scaleFactor;
  node["minNeighbours"] >> minNeighbours;
  node["minWidth"] >> minSize.width;
  node["minHeight"] >> minSize.height;
}
//==============================================================================
void 
write(FileStorage& fs, 
      const string&, 
      const face_tracker_params& x)
{
  x.write(fs);
}
//==============================================================================
void 
read(const FileNode& node, 
     face_tracker_params& x,
     const face_tracker_params& d)
{
  if(node.empty())x = d; else x.read(node);
}
//==============================================================================
face_tracker_params 
load_face_tracker_params(const char* fname)
{
  face_tracker_params x; FileStorage f(fname,FileStorage::READ);
  f["face_tracker_params"] >> x; f.release(); return x;
}
//==============================================================================
void 
save_face_tracker_params(const char* fname,
             const face_tracker_params& x)
{
  FileStorage f(fname,FileStorage::WRITE);
  f << "face_tracker_params" << x; f.release();
}

// 判断rect1是否在rect2里面的功能 
bool isInside(Rect rect1, Rect rect2)
{
	return (rect1 == (rect1&rect2));
}

//==============================================================================
//==============================================================================
//==============================================================================
//============================== face_tracker ==================================
//==============================================================================
//==============================================================================
//==============================================================================
int
face_tracker::
track(const Mat &im, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, const face_tracker_params &p)
{
  //convert image to greyscale
  Mat gray; if(im.channels()==1)gray = im; else cvtColor(im,gray,CV_RGB2GRAY);

  //initialise，为第一帧或下一帧初始化人脸特征
  if(!tracking)
	  points = detector.detect(gray, &storeFaceArea, p.scaleFactor, p.minNeighbours, p.minSize);
  if((int)points.size() != smodel.npts())return 0;

  //fit，通过迭代缩小的搜索范围，估计当前帧中的人脸特征点
  for(int level = 0; level < int(p.ssize.size()); level++)
    points = this->fit(gray,points,p.ssize[level],p.robust,p.itol,p.ftol);

  Rect oldRect = storeFaceArea;
  storeFaceArea.x = points[0].x - 10;
  storeFaceArea.y = points[19].y - 10;
  storeFaceArea.width = points[16].x - points[0].x + 20;
  storeFaceArea.height = points[8].y - points[19].y + 20;
  if (!isInside(storeFaceArea, windowArea)) {
	  storeFaceArea = oldRect;
  }

  *storeFaceRect = storeFaceArea;
  storeLeftEye->x = (points[36].x + points[39].x) / 2 - storeFaceRect->x;
  storeLeftEye->y = (points[36].y + points[39].y) / 2 - storeFaceRect->y;
  storeRightEye->x = (points[42].x + points[45].x) / 2 - storeFaceRect->x;
  storeRightEye->y = (points[42].y + points[45].y) / 2 - storeFaceRect->y;

  //set tracking flag and increment timer
  tracking = true; timer.increment();  return 1;
}

int
face_tracker::
track(const Mat &im, const face_tracker_params &p)
{
	//convert image to greyscale
	Mat gray; if (im.channels() == 1)gray = im; else cvtColor(im, gray, CV_RGB2GRAY);

	//initialise
	if (!tracking)
		points = detector.detect(gray, p.scaleFactor, p.minNeighbours, p.minSize);
	if ((int)points.size() != smodel.npts())return 0;

	//fit
	for (int level = 0; level < int(p.ssize.size()); level++)
		points = this->fit(gray, points, p.ssize[level], p.robust, p.itol, p.ftol);

	//set tracking flag and increment timer
	tracking = true; timer.increment();  return 1;
}
//==============================================================================
void
face_tracker::
draw(Mat &im,
     const Scalar pts_color,
     const Scalar con_color)
{
  int n = points.size(); if(n == 0)return;
  for(int i = 0; i < smodel.C.rows; i++){
    int j = smodel.C.at<int>(i,0),k = smodel.C.at<int>(i,1);
    line(im,points[j],points[k],con_color,1);
  }
  for(int i = 0; i < n; i++)circle(im,points[i],1,pts_color);
}
//==============================================================================
vector<Point2f>
face_tracker::
fit(const Mat &image,				// 当前帧灰度图像
    const vector<Point2f> &init,	// 上一帧人脸特征点集（几何位置）
    const Size ssize,				// 搜索区域大小
    const bool robust,				// 标志位，决定是否采用 robustmodel fitting 流程，应对人脸特征的孤立点
    const int itol,					// robustmodel fitting 迭代上限
    const float ftol)				// 迭代收敛判断阈值
{
  int n = smodel.npts();	// number of points int the shape model
  assert((int(init.size())==n) && (pmodel.n_patches()==n));
  smodel.calc_params(init); vector<Point2f> pts = smodel.calc_shape();

  //find facial features in image around current estimates
  vector<Point2f> peaks = pmodel.calc_peaks(image,pts,ssize);

  //optimise
  if(!robust){
    smodel.calc_params(peaks); //compute shape model parameters        
    pts = smodel.calc_shape(); //update shape
  }else{
    Mat weight(n,1,CV_32F),weight_sort(n,1,CV_32F);
    vector<Point2f> pts_old = pts;
    for(int iter = 0; iter < itol; iter++){
      //compute robust weight
      for(int i = 0; i < n; i++)weight.fl(i) = norm(pts[i] - peaks[i]);
      cv::sort(weight,weight_sort,CV_SORT_EVERY_COLUMN|CV_SORT_ASCENDING);
      double var = 1.4826*weight_sort.fl(n/2); if(var < 0.1)var = 0.1;
      pow(weight,2,weight); weight *= -0.5/(var*var); cv::exp(weight,weight); 

      //compute shape model parameters    
      smodel.calc_params(peaks,weight);
      
      //update shape
      pts = smodel.calc_shape();
      
      //check for convergence
      float v = 0; for(int i = 0; i < n; i++)v += norm(pts[i]-pts_old[i]);
      if(v < ftol)break; else pts_old = pts;
    }
  }return pts;
}
//==============================================================================
void 
face_tracker::
write(FileStorage &fs) const
{
  assert(fs.isOpened()); 
  fs << "{"
     << "detector" << detector
     << "smodel"   << smodel
     << "pmodel"   << pmodel
     << "}";
}
//==============================================================================
void 
face_tracker::
read(const FileNode& node)
{
  assert(node.type() == FileNode::MAP);
  node["detector"] >> detector;
  node["smodel"]   >> smodel;
  node["pmodel"]   >> pmodel;
}
//==============================================================================
