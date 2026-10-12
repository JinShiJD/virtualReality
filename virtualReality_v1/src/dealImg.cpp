#include "../include/dealImg.h"

// using namespace 放在 .cpp 内,
// 仅在本编译单元生效, 不会污染其他翻译单元,
// 从而避免与 windows.h 中 ACCESS_MASK 等类型冲突。
using namespace std;
using namespace cv;




void DealImg::TAGC_Enhancement(const cv::Mat& src, cv::Mat& dst)
{
	CV_Assert(src.type() == CV_8UC3);	// 确保输入图像是3通道的8位图像
	Mat srcFloat;
	src.convertTo(srcFloat, CV_32FC3, 1.0 / 255.0);		// 转换为浮点数并归一化到[0,1]
	dst.create(src.size(), src.type());

	for (int y = 0; y < srcFloat.rows; y++)
	{
		const float* linePS = srcFloat.ptr<float>(y);
		uchar* linePD = dst.ptr<uchar>(y);
		for (int x = 0; x < srcFloat.cols; x++)
		{
			float blue = linePS[0];
			float green = linePS[1];
			float red = linePS[2];

			float L = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
			float A = (blue + red + green) / 3.0f;
			float Gamma = 5.0f + (0.5f - L) * (1.0f - A) - 2.0f * L;

			linePD[0] = cvRound(std::pow(blue, 2.0f / Gamma) * 255.0f);
			linePD[1] = cvRound(std::pow(green, 2.0f / Gamma) * 255.0f);
			linePD[2] = cvRound(std::pow(red, 2.0f / Gamma) * 255.0f);

			linePS += 3;
			linePD += 3;
		}
	}

}
