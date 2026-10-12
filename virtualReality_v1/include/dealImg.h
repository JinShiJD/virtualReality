#ifndef  DEALIMG_H
#define DEALIMG_H



/*
* 输入：图像(图片/摄像头)
*
* 功能：通过调用函数可对 图片/摄像头 做图像处理
* ==>> 因此对应图像处理会有两个函数: A：一个是对图像处理的函数
*									B：一个是对图片处理的函数
*
* 输出：视频对应处理后视频输出，图片对应处理后的图片输出
*/

#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

// 注意:
//   头文件中禁止使用 `using namespace xxx;`,
//   否则会污染所有包含本头文件的翻译单元,
//   与 windows.h (被 onnxruntime_cxx_api.h 引入) 中的 ACCESS_MASK 等
//   Windows 类型冲突, 触发 C2872 "不明确的符号" 错误。
//   using namespace 已移至 dealImg.cpp 中。


class DealImg
{
public:
	//图像增强
	void TAGC_Enhancement(const cv::Mat& src, cv::Mat& dst);


private:
};

#endif // ! DEALIMG_H

