#include <cv.h>  
#include <highgui.h>  
#include <ml.h>  
#include <cxcore.h>  

#include <io.h>
#include <fstream>
#include <iostream>  
using namespace std;

#include <opencv2\opencv.hpp>  
#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\contrib\contrib.hpp>  
using namespace cv;

float featureScaler;       //used to calculate local feature
float innerBrowRaiser;     //AU 1
float outerBrowRaiser;     //AU 2
float browLower;           //AU 4
float upperLidRaiser;      //AU 5
float lidTightener;        //AU 7
float noseWrinkler;        //AU 9
float lipCornerPull;       //AU 12
float lipCornerDepress;    //AU 15
float lowerLipDepress;     //AU 16
float lipStretch;          //AU 20
float lipTightener;        //AU 23
float jawDrop;             //AU 26

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

float getDist(cv::Point p1, cv::Point p2)
{
	float r = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
	return r;
}

float getDistX(cv::Point p1, cv::Point p2)
{
	float r = abs(p1.x - p2.x);
	return r;
}

float getDistY(cv::Point p1, cv::Point p2)
{
	float r = abs(p1.y - p2.y);
	return r;
}

/*********************************
函数名：  PrintMat(CvMat *matrix)
函数输入：matrix指针 数据类型opencv规定的任意一个
函数作用：在屏幕上打印矩阵
**********************************/
void PrintMat(Mat *matrix, bool save_or_show = false, FILE *fp = NULL)
{
	int i = 0;
	int j = 0;
	for (i = 0; i < matrix->rows; i++)//行  
	{
		if (save_or_show)
		{
			fprintf(fp, "\n");
		}
		else
		{
			printf("\n");
		}

		for (j = 0; j < matrix->cols; j++)//列  
		{
			if (save_or_show)
			{
				fprintf(fp, "%9.3f ", (float)cvGetReal2D(matrix, i, j));
			}
			else
			{
				printf("%9.3f ", (float)cvGetReal2D(matrix, i, j));
			}
		}
	}
}
//*****************************  

void trainEmotion() 
{
	string labelsfile = "D:/Workspaces/VS2013/TEST/NonRigidFaceTracking/NonRigidFaceTracking/CK/labels/";
	string landmarksfile = "D:/Workspaces/VS2013/TEST/NonRigidFaceTracking/NonRigidFaceTracking/CK/landmarks/";
	float labeltoint[8] = { 0, 2, 0, 3, -3, 1, -1, -2 };

	float trainData[327][11];
	float labels[327];

	vector<string> fileNames;
	GetAllFiles(labelsfile, fileNames);
	cout << "GetAllFiles finished." << endl;

	for (int i = 0; i < fileNames.size(); i++)
	{
		string landmarkssstr = landmarksfile + fileNames[i].substr(0, 17) + ".txt";
		ifstream file(landmarkssstr.c_str());
		if (!file.is_open()){
			cerr << "Failed opening " << fileNames[i] << " for reading!" << endl;
			return;
		}
		float x, y;
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

		//assign value of nodes
		trainData[i][0] = innerBrowRaiser;
		trainData[i][1] = outerBrowRaiser;
		trainData[i][2] = browLower;
		trainData[i][3] = upperLidRaiser;
		trainData[i][4] = lidTightener;
		trainData[i][5] = noseWrinkler;
		trainData[i][6] = lipCornerPull;
		trainData[i][7] = lowerLipDepress;
		trainData[i][8] = lipStretch;
		trainData[i][9] = lipTightener;
		trainData[i][10] = jawDrop;


		// 读取标签
		string labelsstr = labelsfile + fileNames[i];
		ifstream lfile(labelsstr.c_str());
		if (!lfile.is_open()){
			cerr << "Failed opening " << fileNames[i] << " for reading!" << endl; return;
		}
		float label;
		lfile >> label;
		labels[i] = labeltoint[(int)label];
	}

	Mat trainingDataMat(327, 11, CV_32FC1, trainData);
	Mat labelsMat(327, 1, CV_32FC1, labels);

	cout << "data has finished." << endl;

	//Set SVM params
	CvSVMParams SVM_params;
	SVM_params.svm_type = CvSVM::C_SVC;
	SVM_params.kernel_type = CvSVM::RBF;
	SVM_params.gamma = 0.1;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, FLT_EPSILON);

	// train the SVM
	CvSVM SVM;
	SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), SVM_params);
	cout << "train finished." << endl;

	SVM.save("emotion.yaml");
	cout << "save finished." << endl;
}

int main()
{
	trainEmotion();

	system("pause");
	return 0;
}