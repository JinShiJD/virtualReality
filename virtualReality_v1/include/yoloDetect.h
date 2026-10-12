#ifndef YOLODETECT_H
#define YOLODETECT_H


#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct DetectObject
{
    int classId = -1;
    float confidence = 0.0f;
    cv::Rect box;
};

class YoloDetect
{
public:
    YoloDetect();
    ~YoloDetect();

    bool init(const std::string& modelPath);
    bool detect(const cv::Mat& processImg, std::vector<DetectObject>& detectResult);
    void release();

    /*************************************************
     * 绘制检测结果
     *
     * image        ：待绘制图像
     * detectResult ：检测结果
     *************************************************/
    void drawResult(cv::Mat& image, const std::vector<DetectObject>& detectResult);

    /*************************************************
     * 加载类别名称
     *
     * classFile：
     * classes.txt 或 coco.names
     *************************************************/
    bool loadClassNames(const std::string& classFile);

    /*************************************************
     * 设置是否输出调试信息
     *
     * true  ：输出详细日志
     * false ：关闭调试输出
     *************************************************/
    void setVerbose(bool enable);

    /*************************************************
     * 获取当前调试状态
     *************************************************/
    bool getVerbose() const;

private:

    bool preprocess(const cv::Mat& processImg, cv::Mat& inputImg);

    bool inference(const cv::Mat& inputImg, std::vector<Ort::Value>& outputTensor);

    bool postprocess(
        const cv::Mat& processImg,
        std::vector<Ort::Value>& outputTensor,
        std::vector<DetectObject>& detectResult);

private:

    /*************************************************
     * ONNX Runtime环境
     *************************************************/
    Ort::Env env;

    /*************************************************
     * ONNX Runtime Session配置
     *************************************************/
    Ort::SessionOptions sessionOptions;

    /*************************************************
     * ONNX模型Session
     *************************************************/
    std::unique_ptr<Ort::Session> session;


    /*************************************************
     * 输入输出节点名称
     *
     * 注意：
     * 使用char*保存
     * 析构时需要释放
     *************************************************/
    std::vector<char*> inputNames;
    std::vector<char*> outputNames;


    /*************************************************
     * 模型初始化状态
     *************************************************/
    bool initialized = false;


    /*************************************************
     * 是否打印调试信息
     *************************************************/
    bool verbose = true;

    /*************************************
     * YOLO模型输入尺寸
     *
     * ONNX:
     *
     * [batch,channel,height,width]
     *
     *************************************/
    int inputWidth = 640;
    int inputHeight = 640;

    /*************************************************
     * LetterBox参数
     *************************************************/
    float scale = 1.0f;

    int padLeft = 0;
    int padTop = 0;
    int padRight = 0;
    int padBottom = 0;


    /*************************************************
     * 阈值参数
     *************************************************/

    float confThreshold = 0.15f; //0.25f;
    float scoreThreshold = 0.15f; //0.25f;
    float nmsThreshold = 0.45f;

    /*************************************************
     * 类别名称列表
     *************************************************/
    std::vector<std::string> classNames;
};




#endif // !YOLO_DETECT_H