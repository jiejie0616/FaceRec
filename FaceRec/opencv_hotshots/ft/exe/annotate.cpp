/*
annotate: annotation tool
*/
#include "opencv_hotshots/ft/ft.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <io.h>
//==============================================================================
class annotate{
public:
	int idx;                       //index of image to annotate
	int pidx;                      //index of point to manipulate
	Mat image;                     //current image to display 
	Mat image_clean;               //clean image to display
	ft_data data;                  //annotation data
	const char* wname;             //display window name
	vector<string> instructions;   //annotation instructions

	annotate(){ wname = "Annotate"; idx = 0; pidx = -1; }

	int
		set_current_image(const int idx = 0){
		if ((idx < 0) || (idx > int(data.imnames.size())))return 0;
		image = data.get_image(idx, 2); return 1;
	}
	void
		set_clean_image(){
		image_clean = image.clone();
	}
	void
		copy_clean_image(){
		image_clean.copyTo(image);
	}
	void
		draw_instructions(){
		if (image.empty())return;
		this->draw_strings(image, instructions);
	}
	void
		draw_points(){
		data.draw_points(image, idx);
	}
	void
		draw_chosen_point(){
		if (pidx >= 0)circle(image, data.points[idx][pidx], 1, CV_RGB(0, 255, 0), 2, CV_AA);
	}
	void
		draw_connections(){
		int m = data.connections.size();
		if (m == 0)this->draw_points();
		else{
			if (data.connections[m - 1][1] < 0){
				int i = data.connections[m - 1][0];
				data.connections[m - 1][1] = i;
				data.draw_connect(image, idx); this->draw_points();
				circle(image, data.points[idx][i], 1, CV_RGB(0, 255, 0), 2, CV_AA);
				data.connections[m - 1][1] = -1;
			}
			else{ data.draw_connect(image, idx); this->draw_points(); }
		}
	}
	void
		draw_symmetry(){
		this->draw_points(); this->draw_connections();
		for (int i = 0; i < int(data.symmetry.size()); i++){
			int j = data.symmetry[i];
			if (j != i){
				circle(image, data.points[idx][i], 1, CV_RGB(255, 255, 0), 2, CV_AA);
				circle(image, data.points[idx][j], 1, CV_RGB(255, 255, 0), 2, CV_AA);
			}
		}
		if (pidx >= 0)circle(image, data.points[idx][pidx], 1, CV_RGB(0, 255, 0), 2, CV_AA);
	}
	void
		set_capture_instructions(){
		instructions.clear();
		instructions.push_back(string("Select expressive frames."));
		instructions.push_back(string("s - use this frame"));
		instructions.push_back(string("q - done"));
	}
	void
		set_pick_points_instructions(){
		instructions.clear();
		instructions.push_back(string("Pick Points"));
		instructions.push_back(string("q - done"));
	}
	void
		set_connectivity_instructions(){
		instructions.clear();
		instructions.push_back(string("Pick Connections"));
		instructions.push_back(string("q - done"));
	}
	void
		set_symmetry_instructions(){
		instructions.clear();
		instructions.push_back(string("Pick Symmetric Points"));
		instructions.push_back(string("q - done"));
	}
	void
		set_move_points_instructions(){
		instructions.clear();
		instructions.push_back(string("Move Points"));
		instructions.push_back(string("p - next image"));
		instructions.push_back(string("o - previous image"));
		instructions.push_back(string("q - done"));
	}
	void
		initialise_symmetry(const int index){
		int n = data.points[index].size(); data.symmetry.resize(n);
		for (int i = 0; i < n; i++)data.symmetry[i] = i;
	}
	void
		replicate_annotations(const int index){
		if ((index < 0) || (index >= int(data.points.size())))return;
		for (int i = 0; i < int(data.points.size()); i++){
			if (i == index)continue;
			data.points[i] = data.points[index];
		}
	}
	int
		find_closest_point(const Point2f p,
		const double thresh = 10.0){
		int n = data.points[idx].size(), imin = -1; double dmin = -1;
		for (int i = 0; i < n; i++){
			double d = norm(p - data.points[idx][i]);
			if ((imin < 0) || (d < dmin)){ imin = i; dmin = d; }
		}
		if ((dmin >= 0) && (dmin < thresh))return imin; else return -1;
	}
protected:
	void
		draw_strings(Mat img,
		const vector<string> &text){
		for (int i = 0; i < int(text.size()); i++)this->draw_string(img, text[i], i + 1);
	}
	void
		draw_string(Mat img,
		const string text,
		const int level)
	{
		Size size = getTextSize(text, FONT_HERSHEY_COMPLEX, 0.6f, 1, NULL);
		putText(img, text, Point(0, level*size.height), FONT_HERSHEY_COMPLEX, 0.6f,
			Scalar::all(0), 1, CV_AA);
		putText(img, text, Point(1, level*size.height + 1), FONT_HERSHEY_COMPLEX, 0.6f,
			Scalar::all(255), 1, CV_AA);
	}
}annotation;
//==============================================================================
void pp_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		annotation.data.points[0].push_back(Point2f(x, y));
		annotation.draw_points(); imshow(annotation.wname, annotation.image);
	}
}
//==============================================================================
void pc_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		int imin = annotation.find_closest_point(Point2f(x, y));
		if (imin >= 0){ //add connection
			int m = annotation.data.connections.size();
			if (m == 0)annotation.data.connections.push_back(Vec2i(imin, -1));
			else{
				if (annotation.data.connections[m - 1][1] < 0)//1st connecting point chosen
					annotation.data.connections[m - 1][1] = imin;
				else annotation.data.connections.push_back(Vec2i(imin, -1));
			}
			annotation.draw_connections();
			imshow(annotation.wname, annotation.image);
		}
	}
}
//==============================================================================
void ps_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		int imin = annotation.find_closest_point(Point2f(x, y));
		if (imin >= 0){
			if (annotation.pidx < 0)annotation.pidx = imin;
			else{
				annotation.data.symmetry[annotation.pidx] = imin;
				annotation.data.symmetry[imin] = annotation.pidx;
				annotation.pidx = -1;
			}
			annotation.draw_symmetry();
			imshow(annotation.wname, annotation.image);
		}
	}
}
//==============================================================================
void mv_MouseCallback(int event, int x, int y, int /*flags*/, void* /*param*/)
{
	if (event == CV_EVENT_LBUTTONDOWN){
		if (annotation.pidx < 0){
			annotation.pidx = annotation.find_closest_point(Point2f(x, y));
		}
		else annotation.pidx = -1;
		annotation.copy_clean_image();
		annotation.draw_connections();
		annotation.draw_chosen_point();
		imshow(annotation.wname, annotation.image);
	}
	else if (event == CV_EVENT_MOUSEMOVE){
		if (annotation.pidx >= 0){
			annotation.data.points[annotation.idx][annotation.pidx] = Point2f(x, y);
			annotation.copy_clean_image();
			annotation.draw_connections();
			annotation.draw_chosen_point();
			imshow(annotation.wname, annotation.image);
		}
	}
}
//==============================================================================
class muct_data{
public:
	string name;
	string index;
	vector<Point2f> points;

	muct_data(string str,
		string muct_dir){
		size_t p1 = 0, p2;

		//set image directory
		string idir = muct_dir; if (idir[idir.length() - 1] != '/')idir += "/";
		idir += "jpg/";

		//get image name
		p2 = str.find(",");
		if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
		name = str.substr(p1, p2 - p1);

		if ((strcmp(name.c_str(), "i434xe-fn") == 0) || //corrupted data
			(name[1] == 'r'))name = "";                //ignore flipped images
		else{
			name = idir + str.substr(p1, p2 - p1) + ".jpg"; p1 = p2 + 1;

			//get index
			p2 = str.find(",", p1);
			if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
			index = str.substr(p1, p2 - p1); p1 = p2 + 1;

			//get points
			for (int i = 0; i < 75; i++){
				p2 = str.find(",", p1);
				if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
				string x = str.substr(p1, p2 - p1); p1 = p2 + 1;
				p2 = str.find(",", p1);
				if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
				string y = str.substr(p1, p2 - p1); p1 = p2 + 1;
				points.push_back(Point2f(atoi(x.c_str()), atoi(y.c_str())));
			}
			p2 = str.find(",", p1);
			if (p2 == string::npos){ cerr << "Invalid MUCT file" << endl; exit(0); }
			string x = str.substr(p1, p2 - p1); p1 = p2 + 1;
			string y = str.substr(p1, str.length() - p1);
			points.push_back(Point2f(atoi(x.c_str()), atoi(y.c_str())));
		}
	}
};
//==============================================================================
bool
parse_help(int argc, char** argv)
{
	for (int i = 1; i < argc; i++){
		string str = argv[i];
		if (str.length() == 2){ if (strcmp(str.c_str(), "-h") == 0)return true; }
		if (str.length() == 6){ if (strcmp(str.c_str(), "--help") == 0)return true; }
	}return false;
}
//==============================================================================
string
parse_odir(int argc, char** argv)
{
	string odir = "data/";
	for (int i = 1; i < argc; i++){
		string str = argv[i];
		if (str.length() != 2)continue;
		if (strcmp(str.c_str(), "-d") == 0){
			if (argc > i + 1){ odir = argv[i + 1]; break; }
		}
	}
	if (odir[odir.length() - 1] != '/')odir += "/";
	return odir;
}
//==============================================================================
int
parse_ifile(int argc,
char** argv,
string& ifile)
{
	for (int i = 1; i < argc; i++){
		string str = argv[i];
		if (str.length() != 2)continue;
		if (strcmp(str.c_str(), "-m") == 0){ //MUCT data
			if (argc > i + 1){ ifile = argv[i + 1]; return 2; }
		}
		if (strcmp(str.c_str(), "-v") == 0){ //video file
			if (argc > i + 1){ ifile = argv[i + 1]; return 1; }
		}
	}
	ifile = ""; return 0;
}

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
//==============================================================================
int main(int argc, char** argv)
{
	//parse cmd line options
	if (parse_help(argc, argv)){
		cout << "usage: ./annotate [-v video] [-m muct_dir] [-d output_dir]"
			<< endl; return 0;
	}
	string odir = parse_odir(argc, argv);
	string ifile; int type = parse_ifile(argc, argv, ifile);
	string fname = odir + "annotations2.yaml"; //file to save annotation data to

	//get data
	namedWindow(annotation.wname);
	if (type == 2){ //MUCT data
// 		string lmfile = ifile + "muct-landmarks/muct76-opencv.csv";
// 		ifstream file(lmfile.c_str());
// 		if (!file.is_open()){
// 			cerr << "Failed opening " << lmfile << " for reading!" << endl; return 0;
// 		}
// 		string str; getline(file, str);
// 		while (!file.eof()){
// 			getline(file, str); if (str.length() == 0)break;
// 			muct_data d(str, ifile); if (d.name.length() == 0)continue;
// 			annotation.data.imnames.push_back(d.name);
// 			annotation.data.points.push_back(d.points);
// 		}
// 		file.close();
// 		annotation.data.rm_incomplete_samples();
		string ptsfile = ifile + "landmarks/";
		vector<string> fileNames;
		GetAllFiles(ptsfile, fileNames);
		for (int i = 0; i < fileNames.size(); ++i) {
			string ptsstr = ptsfile + fileNames[i];
			ifstream file(ptsstr.c_str());
			if (!file.is_open()){
				cerr << "Failed opening " << fileNames[i] << " for reading!" << endl; return 0;
			}
			double x, y;
			vector<Point2f> points;
// 			string str;
// 			getline(file, str);
// 			getline(file, str);
// 			getline(file, str);
			for (int j = 0; j < 68; ++j) {
				file >> x;
				file >> y;
				points.push_back(Point2f(x, y));
			}
			string pngstr = ifile + "images/" + fileNames[i].substr(0, fileNames[i].size() - 4) + ".png";
			annotation.data.imnames.push_back(pngstr);
			annotation.data.points.push_back(points);
			file.close();
		}
		annotation.data.rm_incomplete_samples();
	}
	else{
		//open video stream
		VideoCapture cam;
		if (type == 1)cam.open(ifile); else cam.open(0);
		if (!cam.isOpened()){
			cout << "Failed opening video file." << endl
				<< "usage: ./annotate [-v video] [-m muct_dir] [-d output_dir]"
				<< endl; return 0;
		}

		//选择图像进行标注
		annotation.set_capture_instructions();		// 显示帮助信息
		while (cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999){	// 循环遍历每一帧
			Mat im, img; cam >> im; 
			annotation.image = im.clone();
			annotation.draw_instructions();
			imshow(annotation.wname, annotation.image);		// 显示当前帧
			int c = waitKey(0);		// 等待按键，q 退出，s 选择图像进行标注，其它任意键 下一帧
			if (c == 'q')break;
			else if (c == 's'){
				int idx = annotation.data.imnames.size(); char str[1024];
				if (idx < 10)sprintf(str, "00%d.png", idx);
				else if (idx < 100)sprintf(str, "0%d.png",idx);
				else               sprintf(str, "%d.png", idx);		// 文件名格式 三位整数.png
				imwrite(str, im);		// 保存该帧图像
				annotation.data.imnames.push_back(str);
				cam >> im;				// 显示下一帧
				imshow(annotation.wname, im);
			}
		}
		if (annotation.data.imnames.size() == 0)return 0;
		annotation.data.points.resize(annotation.data.imnames.size());

		// 标注第一幅图像
		setMouseCallback(annotation.wname, pp_MouseCallback, 0);
		annotation.set_pick_points_instructions();	// 显示帮助信息
		annotation.set_current_image(0);		// 选择第一幅图像
		annotation.draw_instructions();
		annotation.idx = 0;
		while (1){			// 在键入 q 之前，鼠标单击标注特征点
			annotation.draw_points();
			imshow(annotation.wname, annotation.image); 
			if (waitKey(0) == 'q')break;
		}
		if (annotation.data.points[0].size() == 0)return 0;
		annotation.replicate_annotations(0);	// 保存特征点位置信息
	}
	save_ft(fname.c_str(), annotation.data);	

	//标注连通性
	setMouseCallback(annotation.wname, pc_MouseCallback, 0);
	annotation.set_connectivity_instructions();	// 帮助信息
	annotation.set_current_image(26);
	annotation.draw_instructions();
	annotation.idx = 26;
	while (1){			// 在键入 q 之前，鼠标单击一组点建立连接
		annotation.draw_connections();
		imshow(annotation.wname, annotation.image); if (waitKey(0) == 'q')break;
	}
	save_ft(fname.c_str(), annotation.data);

	//标注对称性
	setMouseCallback(annotation.wname, ps_MouseCallback, 0);
	annotation.initialise_symmetry(26);
	annotation.set_symmetry_instructions();
	annotation.set_current_image(26);
	annotation.draw_instructions();
	annotation.idx = 26; annotation.pidx = -1;
	while (1){			// 在键入 q 之前，鼠标单击特征点标注对称性
		annotation.draw_symmetry();
		imshow(annotation.wname, annotation.image); if (waitKey(0) == 'q')break;
	}
	save_ft(fname.c_str(), annotation.data);

	//标注剩下的图像
	if (type != 2){
		setMouseCallback(annotation.wname, mv_MouseCallback, 0);
		annotation.set_move_points_instructions();		// 帮助信息
		annotation.idx = 1; annotation.pidx = -1;
		while (1){
			annotation.set_current_image(annotation.idx);
			annotation.draw_instructions();
			annotation.set_clean_image();		// 背景图
			annotation.draw_connections();		// 连线
			imshow(annotation.wname, annotation.image);
			int c = waitKey(0);		// q 退出，p 下一幅图像，o 上一幅图像
			if (c == 'q')break;
			else if (c == 'p'){ annotation.idx++; annotation.pidx = -1; }
			else if (c == 'o'){ annotation.idx--; annotation.pidx = -1; }
			if (annotation.idx < 0)annotation.idx = 0;
			if (annotation.idx >= int(annotation.data.imnames.size()))
				annotation.idx = annotation.data.imnames.size() - 1;
		}
	}
	save_ft(fname.c_str(), annotation.data); 
	destroyWindow("Annotate"); return 0;
}
//==============================================================================