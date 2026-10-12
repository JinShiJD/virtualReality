#include "../include/yoloDetect.h"
#include <cstring>
#include <fstream>

YoloDetect::YoloDetect()
{
    /*************************************
     * 创建ONNX Runtime环境
     *************************************/
    env = Ort::Env(
            ORT_LOGGING_LEVEL_WARNING,
            "YOLO11");

    /*************************************
     * 默认未初始化
     *************************************/
    initialized = false;

    /*************************************
    * 默认关闭调试输出
    *************************************/
    verbose = false;

}

YoloDetect::~YoloDetect()
{
    release();
}


void YoloDetect::release()
{

    /*************************************
     * 释放输入节点名称
     *************************************/
    for (auto& name : inputNames)
    {
        if (name != nullptr)
        {
            delete[] name;
            name = nullptr;
        }
    }

    /*************************************
     * 释放输出节点名称
     *************************************/
    for (auto& name : outputNames)
    {
        if (name != nullptr)
        {
            delete[] name;
            name = nullptr;
        }
    }


    /*************************************
     * 清空vector
     *************************************/
    inputNames.clear();
    outputNames.clear();

    /*************************************
     * 释放Session
     *
     * unique_ptr自动释放
     *************************************/
    session.reset();

    /*************************************
     * 标记未初始化
     *************************************/
    initialized = false;

}

void YoloDetect::setVerbose(bool enable)
{
    verbose = enable;
}

bool YoloDetect::getVerbose() const
{
    return verbose;
}


bool YoloDetect::init(const std::string& modelPath)
{
    if (modelPath.empty())
    {
        std::cout << "Model path is empty!" << std::endl;
        return false;
    }

    try
    {
        /*********************************
        * 如果之前加载过模型
        * 先释放
        *********************************/
        release();

        //==============================
        // Session参数
        //==============================
        sessionOptions.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        sessionOptions.SetIntraOpNumThreads(4);

        //==============================
        // 创建Session
        //==============================
#ifdef _WIN32

        std::wstring modelPathW(
            modelPath.begin(),
            modelPath.end());

        session = std::make_unique<Ort::Session>(
            env,
            modelPathW.c_str(),
            sessionOptions);

#else

        session = std::make_unique<Ort::Session>(
            env,
            modelPath.c_str(),
            sessionOptions);

#endif

        Ort::AllocatorWithDefaultOptions allocator;

        //==============================
        // 输入节点
        //==============================
        inputNames.clear();

        size_t inputCount = session->GetInputCount();

        for (size_t i = 0; i < inputCount; i++)
        {
            Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(i, allocator);

            //inputNames.push_back( _strdup(inputName.get()));
#ifdef _WIN32

            inputNames.push_back(_strdup(inputName.get()));

#else

            inputNames.push_back(strdup(inputName.get()));

#endif
        }

        //==============================
        // 输出节点
        //==============================
        outputNames.clear();

        size_t outputCount = session->GetOutputCount();

        for (size_t i = 0; i < outputCount; i++)
        {
            Ort::AllocatedStringPtr outputName =
                session->GetOutputNameAllocated(i, allocator);

            outputNames.push_back(_strdup(outputName.get()));
        }

        //==============================
        // 获取模型输入尺寸
        //==============================
        Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);

        auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> inputShape = tensorInfo.GetShape();

        if (inputShape.size() == 4)
        {

            if (inputShape[1] != 3)
            {
                std::cout
                    << "Warning: Model channel is not 3!"
                    << std::endl;
            }


            if (inputShape[2] > 0 && inputShape[3] > 0)
            {
                inputHeight = static_cast<int>(inputShape[2]);
                inputWidth = static_cast<int>(inputShape[3]);
            }
            else
            {
                std::cout << "Invalid input size!" << std::endl;
                return false;
            }
        }
        else
        {
            std::cout<< "Unsupported input shape!" << std::endl;
            return false;
        }

        //==============================
        // 输出信息
        //==============================
        std::cout << "====================================" << std::endl;

        std::cout << "YOLO11 Model Load Success." << std::endl;
        std::cout << "Input Size : " << inputWidth << " x " << inputHeight << std::endl;
        std::cout << "Input Count : " << inputNames.size() << std::endl;
        std::cout << "Output Count : " << outputNames.size() << std::endl;

        std::cout << "====================================" << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cout << "ONNXRuntime Error : " << e.what() << std::endl;

        return false;
    }

    initialized = true;

    return true;
}

/*
* 图像预处理：
* 调整图像大小
*/ 
bool YoloDetect::preprocess(const cv::Mat& processImg, cv::Mat& inputImg)
{
    if (processImg.empty())
    {
        std::cout << "Input image is empty!" << std::endl;
        return false;
    }

    int srcWidth = processImg.cols;
    int srcHeight = processImg.rows;

    scale = std::min(
        static_cast<float>(inputWidth) / srcWidth,
        static_cast<float>(inputHeight) / srcHeight);

    int resizeWidth = static_cast<int>(srcWidth * scale);

    int resizeHeight = static_cast<int>(srcHeight * scale);

    cv::Mat resizeImg;

    cv::resize(
        processImg,
        resizeImg,
        cv::Size(resizeWidth, resizeHeight));

    //==============================
    // 计算Padding
    //==============================
    int padW = inputWidth - resizeWidth;
    int padH = inputHeight - resizeHeight;
    padLeft = padW / 2;
    padRight = padW - padLeft;
    padTop = padH / 2;
    padBottom = padH - padTop;

    cv::copyMakeBorder(
        resizeImg,
        inputImg,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

    cv::cvtColor(
        inputImg,
        inputImg,
        cv::COLOR_BGR2RGB);

    inputImg.convertTo(
        inputImg,
        CV_32FC3,
        1.0 / 255.0);

    if (!inputImg.isContinuous())
    {
        inputImg = inputImg.clone();
    }

    return true;
}

bool YoloDetect::inference(const cv::Mat& inputImg, std::vector<Ort::Value>& outputTensor)
{
    //==============================
    // 输入图像检查
    //==============================
    if (inputImg.empty())
    {
        std::cout << "Input image is empty!" << std::endl;
        return false;
    }

    //==============================
    // HWC -> CHW
    //==============================
    std::vector<float> inputTensorValues;

    inputTensorValues.resize(3 * inputHeight * inputWidth);

    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; i++)
    {
        channels[i] = cv::Mat(
            inputHeight,
            inputWidth,
            CV_32FC1,
            inputTensorValues.data() + i * inputHeight * inputWidth);
    }

    cv::split(inputImg, channels);

    //==============================
    // 创建Tensor Shape
    //==============================
    std::vector<int64_t> inputShape =
    {
        1,
        3,
        inputHeight,
        inputWidth
    };

    //==============================
    // 创建CPU内存信息
    //==============================
    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

    //==============================
    // 创建输入Tensor
    //==============================
    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size());

    //==============================
    // 开始推理
    //==============================
    try
    {

        outputTensor =
            session->Run(
                Ort::RunOptions{ nullptr },
                inputNames.data(),
                &inputTensor,
                inputNames.size(),
                outputNames.data(),
                outputNames.size());

    }
    catch (const Ort::Exception& e)
    {

        std::cout
            << "ONNX Runtime Error:"
            << e.what()
            << std::endl;


        return false;

    }

    return true;
}

/*
* 图像后处理：
* 非极大值抑制。用于去除重叠的检测框，只保留最好的那个。
*/
bool YoloDetect::postprocess(const cv::Mat& processImg,std::vector<Ort::Value>& outputTensor,
                             std::vector<DetectObject>& detectResult)
{
    /*************************************************
     * 清空上一帧检测结果
     *************************************************/
    detectResult.clear();

    /*************************************************
     * 检查输出Tensor
     *************************************************/
    if (outputTensor.empty())
    {
        std::cout << "Output tensor is empty!" << std::endl;
        return false;
    }

    /*************************************************
     * YOLO11 Detect只有一个输出Tensor
     *************************************************/
    Ort::Value& tensor = outputTensor[0];

    /*************************************************
     * 获取Tensor信息
     *************************************************/
    Ort::TensorTypeAndShapeInfo tensorInfo =
        tensor.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> outputShape =
        tensorInfo.GetShape();

    if (verbose)
    {
        std::cout << "Output Shape : ";

        for (size_t i = 0; i < outputShape.size(); i++)
        {
            std::cout << outputShape[i] << " ";
        }

        std::cout << std::endl;
    }

    /*************************************************
     * YOLO Detect输出必须是三维
     *************************************************/
    if (outputShape.size() != 3)
    {
        std::cout
            << "Unsupported output shape!"
            << std::endl;

        return false;
    }

    /*************************************************
     * 获取输出数据指针
     *************************************************/
    float* outputData =
        tensor.GetTensorMutableData<float>();

    if (outputData == nullptr)
    {
        std::cout
            << "Output data is nullptr!"
            << std::endl;

        return false;
    }

    /*************************************************
     * 判断输出布局
     *
     * 支持：
     * 1×84×8400
     * 1×8400×84
     *************************************************/
    bool transpose = false;

    int boxCount = 0;

    int attributeCount = 0;

    if (outputShape[1] < outputShape[2])
    {
        transpose = true;

        attributeCount =
            static_cast<int>(outputShape[1]);

        boxCount =
            static_cast<int>(outputShape[2]);
    }
    else
    {
        transpose = false;

        boxCount =
            static_cast<int>(outputShape[1]);

        attributeCount =
            static_cast<int>(outputShape[2]);
    }

    if (verbose)
    {
        std::cout
            << "Box Count : "
            << boxCount
            << std::endl;

        std::cout
            << "Attribute Count : "
            << attributeCount
            << std::endl;

        std::cout
            << "Transpose : "
            << std::boolalpha
            << transpose
            << std::endl;
    }

    /*************************************************
     * 保存所有候选框
     *************************************************/
    std::vector<cv::Rect> boxes;

    std::vector<float> scores;

    std::vector<int> classIds;

    boxes.reserve(boxCount);
    scores.reserve(boxCount);
    classIds.reserve(boxCount);

    /*************************************************
     * 遍历所有预测框
     *************************************************/
    for (int i = 0; i < boxCount; i++)
    {
        float cx;
        float cy;
        float w;
        float h;

        if (transpose)
        {
            cx = outputData[0 * boxCount + i];
            cy = outputData[1 * boxCount + i];
            w = outputData[2 * boxCount + i];
            h = outputData[3 * boxCount + i];
        }
        else
        {
            cx = outputData[i * attributeCount + 0];
            cy = outputData[i * attributeCount + 1];
            w = outputData[i * attributeCount + 2];
            h = outputData[i * attributeCount + 3];
        }

        /*************************************************
         * 查找最大类别
         *************************************************/
        float maxScore = 0.0f;

        int maxClassId = -1;

        int classCount = attributeCount - 4;

        for (int c = 0; c < classCount; c++)
        {
            float score;

            if (transpose)
            {
                score =
                    outputData[(c + 4) * boxCount + i];
            }
            else
            {
                score =
                    outputData[i * attributeCount + c + 4];
            }

            if (score > maxScore)
            {
                maxScore = score;
                maxClassId = c;
            }
        }

        /*************************************************
         * 置信度过滤
         *************************************************/
        if (maxScore < confThreshold)
        {
            continue;
        }

        /*************************************************
         * 中心点转左上角
         *************************************************/
        float left = cx - w * 0.5f;
        float top = cy - h * 0.5f;

        /*************************************************
         * 去除LetterBox Padding
         *************************************************/
        left -= static_cast<float>(padLeft);
        top -= static_cast<float>(padTop);

        /*************************************************
         * 恢复原图坐标
         *************************************************/
        left /= scale;
        top /= scale;
        w /= scale;
        h /= scale;

        cv::Rect box(
            static_cast<int>(left),
            static_cast<int>(top),
            static_cast<int>(w),
            static_cast<int>(h));

        /*************************************************
         * 裁剪到图像范围
         *************************************************/
        box &= cv::Rect(
            0,
            0,
            processImg.cols,
            processImg.rows);

        if (box.width <= 0 ||
            box.height <= 0)
        {
            continue;
        }

        boxes.push_back(box);
        scores.push_back(maxScore);
        classIds.push_back(maxClassId);
    }

    /*************************************************
     * 如果没有检测框
     *************************************************/
    if (boxes.empty())
    {
        return true;
    }

    /*************************************************
     * NMS
     *************************************************/
    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        scores,
        scoreThreshold,
        nmsThreshold,
        indices);

    /*************************************************
     * 保存最终结果
     *************************************************/
    detectResult.reserve(indices.size());

    for (int index : indices)
    {
        DetectObject object;

        object.classId = classIds[index];

        object.confidence = scores[index];

        object.box = boxes[index];

        detectResult.push_back(object);
    }

    if (verbose)
    {
        std::cout
            << "Detect Object Count : "
            << detectResult.size()
            << std::endl;
    }

    return true;
}

bool YoloDetect::detect(const cv::Mat& processImg, std::vector<DetectObject>& detectResult)
{
    /*********************************
     * 检查模型状态
     *********************************/
    if (!initialized)
    {
        std::cout
            << "YOLO model is not initialized!"
            << std::endl;

        return false;
    }


    /*********************************
     * 检查输入
     *********************************/
    if (processImg.empty())
    {
        std::cout
            << "Input image empty!"
            << std::endl;

        return false;
    }

    /*************************************************
     * 第一步：
     * 检查输入图像
     *************************************************/
    if (processImg.empty())
    {
        std::cout
            << "Detect input image is empty!"
            << std::endl;

        return false;
    }


    /*************************************************
     * 第二步：
     * 图像预处理
     *
     * processImg
     *
     *        ↓
     *
     * inputImg
     *
     * 完成：
     * BGR->RGB
     * Resize
     * LetterBox
     * Normalize
     *************************************************/
    cv::Mat inputImg;

    if (!preprocess(processImg,inputImg))
    {
        std::cout<< "Preprocess failed!"<< std::endl;
        return false;
    }


    /*************************************************
     * 第三步：
     * ONNX Runtime 推理
     *
     * inputImg
     *
     *        ↓
     *
     * outputTensor
     *************************************************/
    std::vector<Ort::Value> outputTensor;


    if (!inference(
        inputImg,
        outputTensor))
    {
        std::cout
            << "Inference failed!"
            << std::endl;

        return false;
    }

    /*************************************************
     * 第四步：
     * YOLO输出后处理
     *
     * outputTensor
     *
     *        ↓
     *
     * detectResult
     *
     * 完成：
     * 解码
     * 分类
     * 坐标恢复
     * NMS
     *************************************************/
    if (!postprocess(
        processImg,
        outputTensor,
        detectResult))
    {
        std::cout
            << "Postprocess failed!"
            << std::endl;

        return false;
    }

    /*************************************************
     * 第五步：
     * 返回检测成功
     *************************************************/
    return true;
}

// 在图像上绘制矩形框
void YoloDetect::drawResult(cv::Mat& image, const std::vector<DetectObject>& detectResult)
{
    /*************************************************
     * 输入检查
     *************************************************/
    if (image.empty())
    {
        std::cout
            << "Draw image is empty!"
            << std::endl;

        return;
    }

    /*************************************************
     * 没有检测目标
     *************************************************/
    if (detectResult.empty())
    {
        return;
    }

    /*************************************************
     * 遍历所有检测目标
     *************************************************/
    for (const auto& object : detectResult)
    {
        /*********************************************
         * 绘制矩形框
         *********************************************/
        cv::rectangle(
            image,
            object.box,
            cv::Scalar(0, 255, 0),
            2);

        /*********************************************
         * 构造显示文字
         *********************************************/
        std::string label;

        if (object.classId >= 0 && object.classId < static_cast<int>(classNames.size()))
        {
            label = classNames[object.classId];
        }
        else
        {
            label = std::to_string(object.classId);
        }

        label += cv::format(" %.2f", object.confidence);

        /*********************************************
         * 获取文字尺寸
         *********************************************/
        int baseline = 0;

        cv::Size textSize =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseline);

        /*********************************************
         * 文字背景区域
         *********************************************/
        int x = object.box.x;

        int y = object.box.y - textSize.height - 5;

        if (y < 0)
        {
            y = 0;
        }

        cv::rectangle(
            image,
            cv::Rect(
                x,
                y,
                textSize.width + 4,
                textSize.height + 6),
            cv::Scalar(0, 255, 0),
            cv::FILLED);

        /*********************************************
         * 绘制文字
         *********************************************/
        cv::putText(
            image,
            label,
            cv::Point(
                x + 2,
                y + textSize.height + 1),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 0),
            1);
    }
}

bool YoloDetect::loadClassNames(const std::string& classFile)
{
    /*************************************************
     * 清空旧类别
     *************************************************/
    classNames.clear();

    /*************************************************
     * 打开文件
     *************************************************/
    std::ifstream file(classFile);

    if (!file.is_open())
    {
        std::cout
            << "Open class file failed : "
            << classFile
            << std::endl;

        return false;
    }

    /*************************************************
     * 逐行读取
     *************************************************/
    std::string line;

    while (std::getline(file, line))
    {
        /*********************************************
         * 去掉空行
         *********************************************/
        if (line.empty())
        {
            continue;
        }

        classNames.push_back(line);
    }

    file.close();

    /*************************************************
     * 调试输出
     *************************************************/
    if (verbose)
    {
        std::cout
            << "Load class names : "
            << classNames.size()
            << std::endl;
    }

    return true;
}