/*
  patch_model: Correlation patch experts
*/
#include "opencv_hotshots/ft/patch_model.h"
#include "opencv_hotshots/ft/ft.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "stdio.h"      // For 'sprintf()'

#define fl at<float>
//==============================================================================
Mat
patch_model::
convert_image(const Mat &im)
{
  Mat I; 
  if(im.channels() == 1){
    if(im.type() != CV_32F)im.convertTo(I,CV_32F); 
    else I = im;
  }else{
    if(im.channels() == 3){
      Mat img; cvtColor(im,img,CV_RGB2GRAY);
      if(img.type() != CV_32F)img.convertTo(I,CV_32F); 
      else I = img;
    }else{cout << "Unsupported image type!" << endl; abort();}
  }
  I += 1.0; log(I,I); return I;
}
//==============================================================================
Mat 
patch_model::
calc_response(const Mat &im,const bool sum2one)
{
  Mat I = this->convert_image(im);
  Mat res; matchTemplate(I,P,res,CV_TM_CCOEFF_NORMED); 
  if(sum2one){
    normalize(res,res,0,1,NORM_MINMAX); res /= sum(res)[0];
  }return res;
}
//==============================================================================
void 
patch_model::
train(const vector<Mat> &images,	// 包含多个样本图像的矩阵向量
      const Size psize,				// 团块模型窗口的大小
      const float var,				// 手工标注错误的方差（生成理想图像时使用）
      const float lambda,			// 调整的参数（调整上一次得到的团块模型的大小）
      const float mu_init,			// 初始步长（构建梯度下降法求团块模型时的更新速率）
      const int nsamples,			// 随机选取的样本数量（梯度下降算法迭代的次数）
      const bool visi)				// 训练过程是否可观察标志
{
  int N = images.size(),n = psize.width*psize.height;

  // 生成服从高斯分布的理想反馈图像 F
  Size wsize = images[0].size();
  if((wsize.width < psize.width) || (wsize.height < psize.height)){
    cerr << "Invalid image size < patch size!" << endl; throw std::exception();
  }
  // 设置反馈图像大小
  int dx = wsize.width-psize.width,dy = wsize.height-psize.height;
  Mat F(dy,dx,CV_32F);
  for(int y = 0; y < dy; y++){   float vy = (dy-1)/2 - y;
    for(int x = 0; x < dx; x++){ float vx = (dx-1)/2 - x;
	  // 生成函数
      F.fl(y,x) = exp(-0.5*(vx*vx+vy*vy)/var);
    }
  }
  // 归一化处理
  normalize(F,F,0,1,NORM_MINMAX);

  //allocate memory
  Mat I(wsize.height,wsize.width,CV_32F);				// 被选中的样本灰度图像
  Mat dP(psize.height,psize.width,CV_32F);				// 目标函数的偏导数，大小同团块模型
  Mat O = Mat::ones(psize.height,psize.width,CV_32F)/n;	// 生成团块模型的归一化模板
  P = Mat::zeros(psize.height,psize.width,CV_32F);		// 团块模型

  // 利用随机梯度下降法求最优团块模型
  RNG rn(getTickCount()); 
  // 给定初始更新速率
  double mu=mu_init,step=pow(1e-8/mu_init,1.0/nsamples);
  for(int sample = 0; sample < nsamples; sample++){ 
	int i = rn.uniform(0,N);							// i 为随机选中的样本图像标记
    I = this->convert_image(images[i]); dP = 0.0;		// 将图像转换为灰度图
    for(int y = 0; y < dy; y++){
      for(int x = 0; x < dx; x++){
    Mat Wi = I(Rect(x,y,psize.width,psize.height)).clone();
    Wi -= Wi.dot(O); normalize(Wi,Wi);
	// 计算目标函数的偏导数 D 
    dP += (F.fl(y,x) - P.dot(Wi))*Wi;
      }
    }  
	// 更新团块模型 P
    P += mu*(dP - lambda*P); mu *= step;
    if(visi){
      Mat R; 
	  matchTemplate(I,P,R,CV_TM_CCOEFF_NORMED);	// 在样本图像上寻找与团块模型匹配的区域
      Mat PP; normalize(P,PP,0,1,NORM_MINMAX);
      normalize(dP,dP,0,1,NORM_MINMAX);
      normalize(R,R,0,1,NORM_MINMAX);
      imshow("P",PP);			// 归一化的团块模型
	  imshow("dP",dP);			// 归一化的目标函数偏导数
	  imshow("R",R);			// 与团块模型匹配的区域
      if(waitKey(10) == 27)break;
    }
  }return;
}
//==============================================================================
void
patch_model::
write(FileStorage &fs) const
{
  assert(fs.isOpened()); fs << "{" << "P"  << P << "}";
}  
//==============================================================================
void
patch_model::
read(const FileNode& node)
{
  assert(node.type() == FileNode::MAP); node["P"] >> P; 
}
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
void 
patch_models::
train(ft_data &data,
      const vector<Point2f> &ref,
      const Size psize,
      const Size ssize,
      const bool mirror,
      const float var,
      const float lambda,
      const float mu_init,
      const int nsamples,
      const bool visi)
{
  //set reference shape
  int n = ref.size(); reference = Mat(ref).reshape(1,2*n);
  Size wsize = psize + ssize;	// wsize 为归一化的样本图像大小

  //train each patch model in turn
  // n 个特征点将对应 n 个团块
  patches.resize(n);
  for(int i = 0; i < n; i++){
    if(visi)cout << "training patch " << i << "..." << endl;
    vector<Mat> images(0);
    for(int j = 0; j < data.n_images(); j++){	// 遍历所有样本图像
      Mat im = data.get_image(j,0); 
      vector<Point2f> p = data.get_points(j,false);	// 获取手工标注的样本点
      Mat pt = Mat(p).reshape(1,2*n);
      Mat S = this->calc_simil(pt),A(2,3,CV_32F);	// 计算样本点到参考模型的变化矩阵
      A.fl(0,0) = S.fl(0,0); A.fl(0,1) = S.fl(0,1);	// 构造仿射变换矩阵
      A.fl(1,0) = S.fl(1,0); A.fl(1,1) = S.fl(1,1);
      A.fl(0,2) = pt.fl(2*i  ) - 
    (A.fl(0,0) * (wsize.width-1)/2 + A.fl(0,1)*(wsize.height-1)/2);
      A.fl(1,2) = pt.fl(2*i+1) - 
    (A.fl(1,0) * (wsize.width-1)/2 + A.fl(1,1)*(wsize.height-1)/2);
      Mat I; warpAffine(im,I,A,wsize,INTER_LINEAR+WARP_INVERSE_MAP); // 对样本进行仿射变换
      images.push_back(I);
      if(mirror){
    im = data.get_image(j,1); 
    p = data.get_points(j,true);
    pt = Mat(p).reshape(1,2*n);
    S = this->calc_simil(pt);
    A.fl(0,0) = S.fl(0,0); A.fl(0,1) = S.fl(0,1);
    A.fl(1,0) = S.fl(1,0); A.fl(1,1) = S.fl(1,1);
    A.fl(0,2) = pt.fl(2*i  ) - 
      (A.fl(0,0) * (wsize.width-1)/2 + A.fl(0,1)*(wsize.height-1)/2);
    A.fl(1,2) = pt.fl(2*i+1) - 
      (A.fl(1,0) * (wsize.width-1)/2 + A.fl(1,1)*(wsize.height-1)/2);
    warpAffine(im,I,A,wsize,INTER_LINEAR+WARP_INVERSE_MAP);
    images.push_back(I);
      }
    }
    patches[i].train(images,psize,var,lambda,mu_init,nsamples,visi);	// 从样本图像中训练团块模型
  }
}
//==============================================================================
vector<Point2f> 
patch_models::
calc_peaks(const Mat &im,				// 当前包含人脸的灰度图像
       const vector<Point2f> &points,	// 前一帧估计的人脸特征点集在人脸子空间投影坐标集合
       const Size ssize)				// 搜索区域大小
{	
  int n = points.size(); assert(n == int(patches.size()));
  Mat pt = Mat(points).reshape(1,2*n);
  Mat S = this->calc_simil(pt);		// 计算当前点集到人脸参考模型的变化矩阵
  Mat Si = this->inv_simil(S);		// 对矩阵 S 求逆
  // 人脸子空间坐标经过仿射变换转成图像空间中的坐标
  vector<Point2f> pts = this->apply_simil(Si,points);
  for(int i = 0; i < n; i++){
    Size wsize = ssize + patches[i].patch_size(); Mat A(2,3,CV_32F);     
    A.fl(0,0) = S.fl(0,0); A.fl(0,1) = S.fl(0,1);
    A.fl(1,0) = S.fl(1,0); A.fl(1,1) = S.fl(1,1);
    A.fl(0,2) = pt.fl(2*i  ) - 
      (A.fl(0,0) * (wsize.width-1)/2 + A.fl(0,1)*(wsize.height-1)/2);
    A.fl(1,2) = pt.fl(2*i+1) - 
      (A.fl(1,0) * (wsize.width-1)/2 + A.fl(1,1)*(wsize.height-1)/2);
    Mat I; warpAffine(im,I,A,wsize,INTER_LINEAR+WARP_INVERSE_MAP);
	// 搜索人脸特征的匹配位置
    Mat R = patches[i].calc_response(I,false);
    Point maxLoc; minMaxLoc(R,0,0,0,&maxLoc);
	// 修正人脸特征估计点位置
    pts[i] = Point2f(pts[i].x + maxLoc.x - 0.5*ssize.width,
             pts[i].y + maxLoc.y - 0.5*ssize.height);
  }
  // 再次将图像中的坐标投影到人脸特征子空间中，作为下一帧特征估计点位置
  return this->apply_simil(S,pts);
}
//=============================================================================
vector<Point2f> 
patch_models::
apply_simil(const Mat &S,
        const vector<Point2f> &points)
{
  int n = points.size();
  vector<Point2f> p(n);
  for(int i = 0; i < n; i++){
    p[i].x = S.fl(0,0)*points[i].x + S.fl(0,1)*points[i].y + S.fl(0,2);
    p[i].y = S.fl(1,0)*points[i].x + S.fl(1,1)*points[i].y + S.fl(1,2);
  }return p;
}
//=============================================================================
Mat 
patch_models::
inv_simil(const Mat &S)
{
  Mat Si(2,3,CV_32F);
  float d = S.fl(0,0)*S.fl(1,1) - S.fl(1,0)*S.fl(0,1);
  Si.fl(0,0) = S.fl(1,1)/d; Si.fl(0,1) = -S.fl(0,1)/d;
  Si.fl(1,1) = S.fl(0,0)/d; Si.fl(1,0) = -S.fl(1,0)/d;
  Mat Ri = Si(Rect(0,0,2,2));
  Mat t = -Ri*S.col(2),St = Si.col(2); t.copyTo(St); return Si; 
}
//=============================================================================
Mat
patch_models::
calc_simil(const Mat &pts)
{
  //compute translation
  int n = pts.rows/2; 
  // 计算标注点的重心
  float mx = 0,my = 0;
  for(int i = 0; i < n; i++){
    mx += pts.fl(2*i); my += pts.fl(2*i+1);
  }  
  Mat p(2*n,1,CV_32F); mx /= n; my /= n;
  for(int i = 0; i < n; i++){
    p.fl(2*i) = pts.fl(2*i) - mx; p.fl(2*i+1) = pts.fl(2*i+1) - my;
  }
  //compute rotation and scale
  float a=0,b=0,c=0;
  for(int i = 0; i < n; i++){
    a += reference.fl(2*i) * reference.fl(2*i  ) + 
      reference.fl(2*i+1) * reference.fl(2*i+1);
    b += reference.fl(2*i) * p.fl(2*i  ) + reference.fl(2*i+1) * p.fl(2*i+1);
    c += reference.fl(2*i) * p.fl(2*i+1) - reference.fl(2*i+1) * p.fl(2*i  );
  }
  b /= a; c /= a;
  float scale = sqrt(b*b+c*c),theta = atan2(c,b); 
  float sc = scale*cos(theta),ss = scale*sin(theta);
  // 前两列为缩放旋转，最后一列为平移
  return (Mat_<float>(2,3) << sc,-ss,mx,ss,sc,my);	
}
//==============================================================================
void 
patch_models::
write(FileStorage &fs) const
{
  assert(fs.isOpened()); 
  fs << "{" << "reference" << reference;
  fs << "n_patches" << (int)patches.size();
  for(int i = 0; i < int(patches.size()); i++){
    char str[256]; const char* ss;
    sprintf(str,"patch %d",i); ss = str; fs << ss << patches[i];
  }
  fs << "}";
}
//==============================================================================
void 
patch_models::
read(const FileNode& node)
{
  assert(node.type() == FileNode::MAP); 
  node["reference"] >> reference;
  int n; node["n_patches"] >> n; patches.resize(n);
  for(int i = 0; i < n; i++){
    char str[256]; const char* ss;
    sprintf(str,"patch %d",i); ss = str; node[ss] >> patches[i];
  }
}
//==============================================================================
