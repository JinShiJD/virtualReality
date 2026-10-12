#include <filesystem>

//ONNXRuntime 加载yolo
#include<onnxruntime_cxx_api.h>

#include "../include/virtualReality_v1.h"
#include "../include/vrBridge.h"




using namespace std;

int main()
{
	//01_对文件中图片的操作
#if 0 
	vector<string> fileNames;

	glob("D:/vs2019_files01/opencv_algorithm/pic/new/*.bmp", fileNames);
	for (auto fileName : fileNames)
	{
		Mat src = imread(fileName);
		if (src.empty())
		{
			cout << "Failed to load image!" << endl;
			return -1;
		}

		Mat dst;
		DealImg dealImg;
		dealImg.TAGC_Enhancement(src, dst);

		imshow("origin Pic", src);
		imshow("after Pic", dst);
		waitKey(0);
	}
#endif



	// 02_延伸至对视频的处理
#if 0
	VideoCapture cap(0);
	if (!cap.isOpened())	
		return -1;

	Mat frame;
	Mat dst;
	DealImg dealImg;

	//namedWindow("视频暗光处理", WINDOW_AUTOSIZE);
	for (;;)
	{
		cap >> frame;
		if (frame.empty())
			break;

		dealImg.TAGC_Enhancement(frame, dst);

		
		imshow("摄像头原显示", frame);
		imshow("处理后的显示", dst);

		//waitKey(0);
		if (waitKey(33) >= 0) 
			break;
	}
#endif


	//03_添加ONNXRuntime加载yolo模型进行识别
#if 0

	/*************************************************
	 * 创建OpenCV图像处理对象
	 *************************************************/
	DealImg dealImg;

	/*************************************************
	 * 创建YOLO检测对象
	 *************************************************/
	YoloDetect yolo;

	/*************************************************
	 * 开启调试输出（开发阶段建议开启）
	 *************************************************/
	yolo.setVerbose(true);

	/*************************************************
	 * 加载YOLO模型
	 *************************************************/
	//if (!yolo.init("../model/yolo11n.onnx"))
	namespace fs = std::filesystem;
	fs::path curP = fs::current_path();
	std::cout << "当前的文件路径是： " << curP << std::endl;

	if (!yolo.init("../model/yolo11n.onnx"))
	{
		std::cout
			<< "Load YOLO model failed!"
			<< std::endl;

		return -1;
	}

	/*************************************************
	 * 加载类别名称
	 *************************************************/
	if (!yolo.loadClassNames("../model/classes.txt"))
	{
		std::cout
			<< "Load class file failed!"
			<< std::endl;

		return -1;
	}

	std::cout
		<< "System Initialize Success."
		<< std::endl;


	/*************************************************
	* 读取测试图片
	*************************************************/
	cv::Mat srcImg = cv::imread("../img/zidane.jpg");

	if (srcImg.empty())
	{
		std::cout
			<< "Read image failed!"
			<< std::endl;

		return -1;
	}
	/*************************************************
	 * 显示原图
	 *************************************************/
	cv::imshow(
		"Source Image",
		srcImg);

	cv::waitKey(1);


	/*************************************************
	 * OpenCV处理后的图像
	 *************************************************/
	cv::Mat processImg;

	// 处理后图像的显示
	dealImg.TAGC_Enhancement(srcImg, processImg);
	//cv::imshow("Process Image",	processImg);

	//cv::waitKey(1);


	/*************************************************
	 * 检测结果
	 *************************************************/
	std::vector<DetectObject> detectResult;

	/*************************************************
	* YOLO检测
	*************************************************/
	if (!yolo.detect(processImg, detectResult))
	{
		std::cout << "YOLO Detect Failed!" << std::endl;

		return -1;
	}

	/*************************************************
	 * 输出检测目标数量
	 *************************************************/
	std::cout
		<< "Detect Count : "
		<< detectResult.size()
		<< std::endl;


	/*************************************************
	* 绘制检测结果
	*************************************************/
	yolo.drawResult(
		processImg,
		detectResult);

	/*************************************************
	 * 显示检测结果
	 *************************************************/
	cv::imshow(
		"YOLO11 Detect",
		processImg);

	/*************************************************
	 * 输出检测结果
	 *************************************************/
	for (size_t i = 0; i < detectResult.size(); i++)
	{
		const DetectObject& object = detectResult[i];

		std::cout << "Object " << i << std::endl;
		std::cout << "  Class ID   : " << object.classId << std::endl;
		std::cout << "  Confidence : " << object.confidence << std::endl;
		std::cout << "  Box : " << object.box.x << ", " << object.box.y << ", "
			<< object.box.width << ", " << object.box.height << std::endl;
	}


	/*************************************************
	 * 等待按键退出
	 *************************************************/
	cv::waitKey(0);


#endif


	//04 网络数据的测试
#if 0
	
	VRBridge server(8088);
	if (!server.start()) {
		return -1;
	}

	int frameid = 0;
	while (true) {
		// 模拟生成检测数据
		FrameData data;
		data.frameId = frameid++;
		data.imageSize = { 1920, 1080 };

		ObjectInfo obj;
		obj.classId = 0;
		obj.className = "person";
		obj.confidence = 0.95f;
		obj.box = { 100, 200, 80, 160 };
		data.objects.push_back(obj);

		// 发送数据
		server.sendDetectionData(data);

		// 模拟帧率 30fps
		std::this_thread::sleep_for(std::chrono::milliseconds(33));
	}

	server.stop();

#endif





	puts("=================================================");
	cout << "Hello CMake." << endl;
	return 0;
}
