/*
  train_face_tracker: build a face_tracker object from its constituents
*/
#include "opencv_hotshots/ft/ft.hpp"
#include <iostream>
#define fl at<float>
const char* usage = 
"usage: ./train_face_tracker shape_model_file patch_models_file"
" face_detector_file face_tracker_file";
//==============================================================================
bool
parse_help(int argc,char** argv)
{
  for(int i = 1; i < argc; i++){
    string str = argv[i];
    if(str.length() == 2){if(strcmp(str.c_str(),"-h") == 0)return true;}
    if(str.length() == 6){if(strcmp(str.c_str(),"--help") == 0)return true;}
  }return false;
}
//==============================================================================
int main(int argc,char** argv)
{
  //parse cmdline input
  if(parse_help(argc,argv)){ cout << usage << endl; return 0;}
  if(argc < 5){ cout << usage << endl; return 0;}

  //create face tracker model
  face_tracker tracker;
  tracker.smodel = load_ft<shape_model>(argv[1]);		// ���� shape_model ����
  tracker.pmodel = load_ft<patch_models>(argv[2]);		// ���� patch_model ����
  tracker.detector = load_ft<face_detector>(argv[3]);	// ���� face_detector ����
  
  //save face tracker
  save_ft<face_tracker>(argv[4],tracker);				// ���� face_tracker ����
  return 0;
}
//==============================================================================
