#pragma once

// 网络传输

#include <libwebsockets.h>


#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <nlohmann/json.hpp>


// ================== 数据结构定义 ==================

// 对应 JSON 中的 box 对象
struct BoundingBox {
    int x;
    int y;
    int width;
    int height;
};

// 对应 JSON 中的 objects 数组元素
struct ObjectInfo {
    int classId;
    std::string className;
    float confidence;
    BoundingBox box;
};

// 对应 JSON 中的 imageSize
struct ImageSize {
    int width;
    int height;
};

// 顶层 JSON 数据结构
struct FrameData {
    int frameId;
    ImageSize imageSize;
    std::vector<ObjectInfo> objects;
};

// ================== Nlohmann JSON 序列化映射 ==================
// 使用宏定义自动实现 to_json 方法，无需手写序列化代码

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(BoundingBox, x, y, width, height)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ObjectInfo, classId, className, confidence, box)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ImageSize, width, height)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(FrameData, frameId, imageSize, objects)

// ================== VRBridge 类定义 ==================

class VRBridge
{
public:
	char* now() const;	// 获取当前时间


    VRBridge(int port);
    ~VRBridge();

    // 启动服务端（非阻塞，后台运行）
    bool start();

    // 停止服务端
    void stop();

    // 发送检测数据（线程安全）
    void sendDetectionData(const FrameData& data);

private:
    // 内部发送处理
    void broadcastMessage(std::string&& message);

    // 静态回调函数（转发）
    static int wsCallback(struct lws* wsi, enum lws_callback_reasons reason,
        void* user, void* in, size_t len);

    // 实际回调逻辑处理
    int handleCallback(struct lws* wsi, enum lws_callback_reasons reason,
        void* user, void* in, size_t len);

    // 后台服务线程
    void runService();

    int m_port;
    struct lws_context* m_context;
    std::thread m_serviceThread;
    std::atomic<bool> m_running;

    // 客户端列表管理
    std::vector<struct lws*> m_clients;
    std::mutex m_clientMutex;

    // 发送缓冲区
    // 由于检测数据是实时流，我们只保留最新的一帧数据
    std::string m_latestPayload;
    std::mutex m_payloadMutex;

};