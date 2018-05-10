// Use opencv built-in methods to get image filenames of specified folder.  

#include <iostream> 
#include <fstream>
#include <list>
#include <iterator>
#include <vector>
#include <string>
#include <io.h>
using namespace std;

#include "opencv_hotshots\ft\svm.h"
#include <opencv2\opencv.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\contrib\contrib.hpp>  
using namespace cv;

svm_parameter param;
svm_problem prob;
svm_model *svmModel;
list<svm_node*> xList;
list<double>  yList;

double featureScaler;       //used to calculate local feature
double innerBrowRaiser;     //AU 1
double outerBrowRaiser;     //AU 2
double browLower;           //AU 4
double upperLidRaiser;      //AU 5
double lidTightener;        //AU 7
double noseWrinkler;        //AU 9
double lipCornerPull;       //AU 12
double lipCornerDepress;    //AU 15
double lowerLipDepress;     //AU 16
double lipStretch;          //AU 20
double lipTightener;        //AU 23
double jawDrop;             //AU 26

//获取所有的文件名    
void GetAllFiles(string path, vector<string>& files)
{

	long   hFile = 0;
	//文件信息      
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(fileinfo.name);
					GetAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else
			{
				files.push_back(fileinfo.name);
			}

		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}

}

void setParam()
{
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0.1;
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 40;
	param.C = 500;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	// param.probability = 0;
	param.nr_weight = 0;
	param.weight = NULL;
	param.weight_label = NULL;
}

double getDist(cv::Point p1, cv::Point p2)
{
	double r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	return r;
}

double getDistX(cv::Point p1, cv::Point p2)
{
	double r = abs(p1.x - p2.x);
	return r;
}

double getDistY(cv::Point p1, cv::Point p2)
{
	double r = abs(p1.y - p2.y);
	return r;
}

int main(int argc, char* argv[])
{
	string labelsfile = "D:/Workspaces/VS2013/TEST/NonRigidFaceTracking/NonRigidFaceTracking/CK/labels/";
	string landmarksfile = "D:/Workspaces/VS2013/TEST/NonRigidFaceTracking/NonRigidFaceTracking/CK/landmarks/";
	int labeltoint[8] = { 0, 2, 0, 3, -3, 1, -1, -2 };

	vector<string> fileNames;
	GetAllFiles(labelsfile, fileNames);

	double trainData[327][10];

	for (int i = 0; i < fileNames.size(); i++)
	{
		string landmarkssstr = landmarksfile + fileNames[i].substr(0, 17) + ".txt";
		ifstream file(landmarkssstr.c_str());
		if (!file.is_open()){
			cerr << "Failed opening " << fileNames[i] << " for reading!" << endl; return 0;
		}
		double x, y;
		vector<Point2f> cPoints;
		while (!file.eof()) {
			file >> x;
			file >> y;
			cPoints.push_back(Point2f(x, y));
		}
		//assign feature scaler as the width of the face, which does not change in response to different expression
		featureScaler = (getDistX(cPoints[0], cPoints[16]) + getDistX(cPoints[1], cPoints[15]) + getDistX(cPoints[2], cPoints[14])) / 3;
		//assign action unit 1
		innerBrowRaiser = ((getDistY(cPoints[21], cPoints[27]) + getDistY(cPoints[22], cPoints[27])) / 2) / featureScaler;
		//assign action unit 2
		outerBrowRaiser = ((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[26], cPoints[27])) / 2) / featureScaler;
		//assign action unit 4
		browLower = (((getDistY(cPoints[17], cPoints[27]) + getDistY(cPoints[18], cPoints[27]) +
			getDistY(cPoints[19], cPoints[27]) + getDistY(cPoints[20], cPoints[27]) +
			getDistY(cPoints[21], cPoints[27])) / 5 +
			(getDistY(cPoints[22], cPoints[27]) + getDistY(cPoints[23], cPoints[27]) +
			getDistY(cPoints[24], cPoints[27]) + getDistY(cPoints[25], cPoints[27]) +
			getDistY(cPoints[26], cPoints[27])) / 5) / 2) / featureScaler;
		//assign action unit 5
		upperLidRaiser = ((getDistY(cPoints[37], cPoints[27]) + getDistY(cPoints[44], cPoints[27])) / 2) / featureScaler;
		//assign action unit 7
		lidTightener = ((getDistY(cPoints[37], cPoints[41]) + getDistY(cPoints[38], cPoints[40])) / 2 +
			(getDistY(cPoints[43], cPoints[47]) + getDistY(cPoints[44], cPoints[46])) / 2) / featureScaler;
		//assign action unit 9
		noseWrinkler = (getDistY(cPoints[29], cPoints[27]) + getDistY(cPoints[30], cPoints[27])) / featureScaler;
		//assign action unit 12
		lipCornerPull = ((getDistY(cPoints[48], cPoints[33]) + getDistY(cPoints[54], cPoints[33])) / 2) / featureScaler;
		//assign action unit 16
		lowerLipDepress = getDistY(cPoints[57], cPoints[33]) / featureScaler;
		//assign action unit 20
		lipStretch = getDistX(cPoints[48], cPoints[54]) / featureScaler;
		//assign action unit 23
		lipTightener = (getDistY(cPoints[49], cPoints[59]) +
			getDistY(cPoints[50], cPoints[58]) +
			getDistY(cPoints[51], cPoints[57]) +
			getDistY(cPoints[52], cPoints[56]) +
			getDistY(cPoints[53], cPoints[55])) / featureScaler;
		//assign action unit 26
		jawDrop = getDistY(cPoints[8], cPoints[27]) / featureScaler;
		svm_node* node = new svm_node[11 + 1];
		for (int j = 0; j < 11; j++) {
			node[j].index = j + 1;
		}
		node[11].index = -1;

		//assign value of nodes
		node[0].value = innerBrowRaiser;
		node[1].value = outerBrowRaiser;
		node[2].value = browLower;
		node[3].value = upperLidRaiser;
		node[4].value = lidTightener;
		node[5].value = noseWrinkler;
		node[6].value = lipCornerPull;
		node[7].value = lowerLipDepress;
		node[8].value = lipStretch;
		node[9].value = lipTightener;
		node[10].value = jawDrop;

		xList.push_back(node);

		// 读取标签
		string labelsstr = labelsfile + fileNames[i];
		ifstream lfile(labelsstr.c_str());
		if (!lfile.is_open()){
			cerr << "Failed opening " << fileNames[i] << " for reading!" << endl; return 0;
		}
		double label;
		lfile >> label;
		yList.push_back(labeltoint[(int)label]);
	}

	setParam();
	prob.l = fileNames.size();
	prob.x = new svm_node *[prob.l];  //对应的特征向量
	prob.y = new double[prob.l];    //放的是值
	int index = 0;
	while (!xList.empty())
	{
		prob.x[index] = xList.front();
		prob.y[index] = yList.front();
		xList.pop_front();
		yList.pop_front();
		index++;
	}
	svmModel = svm_train(&prob, &param);
	svm_save_model("svm_model1.yaml", svmModel);

	//释放空间
	delete  prob.y;
	delete[] prob.x;
	svm_free_and_destroy_model(&svmModel);

	system("pause");

	return 0;
}