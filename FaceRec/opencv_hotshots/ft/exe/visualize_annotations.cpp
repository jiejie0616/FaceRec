/*
  visualize_annotations: Display annotated data to screen
*/
#include "opencv_hotshots/ft/ft.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//==============================================================================
int main(int argc,char** argv)
{
  // ��������
  string annotation_file = "annotations2.yaml";
  ft_data data = load_ft<ft_data>(annotation_file.c_str());
  if(data.imnames.size() == 0){
    cerr << "Data file does not contain any annotations."<< endl; return 0;
  }
  data.rm_incomplete_samples();

  cout << "n images: " << data.imnames.size() << endl
       << "n points: " << data.symmetry.size() << endl
       << "n connections: " << data.connections.size() << endl;
  // ���ӻ���ע����
  namedWindow("Annotations");
  int index = 0; bool flipped = false;
  while(1){
    Mat image;
    if(flipped)image = data.get_image(index,3);
    else image = data.get_image(index,2);			// ����ͼƬ
    data.draw_connect(image,index,flipped);			// ��ͨ
    data.draw_sym(image,index,flipped);				// �Գ�
    imshow("Annotations",image);
    int c = waitKey(0);			// q �˳���p ��һ�ţ�o ��һ��,f ��ת
    if(c == 'q')break;
    else if(c == 'p')index++;
    else if(c == 'o')index--;
    else if(c == 'f')flipped = !flipped;
    if(index < 0)index = 0;
    else if(index >= int(data.imnames.size()))index = data.imnames.size()-1;
  }
  destroyWindow("Annotations"); return 0;
}
//==============================================================================
