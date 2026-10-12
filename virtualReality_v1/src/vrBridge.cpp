#include "../include/vrBridge.h"
#include <time.h>
#include <cstring>
#include <chrono>
#include <iostream>

// 在函数内或头文件中使用别名是安全的
using json = nlohmann::json;


// 获取当前时间
char * VRBridge::now() const
{
	// 定义一个静态数组用于存储时间字符串，保证函数返回后内存依然有效
	static char buf[32];

	// 获取当前日历时间
	time_t rawTime;
	time(&rawTime); // 等同rawTime = time(NULL);

	// 将time_t时间转换为本地时间结构体
	struct tm* timeInfo;
	timeInfo = localtime(&rawTime);

	snprintf(buf, sizeof(buf), "%04d-%02d-%02d-%02d %02d:%02d:%02d",
		timeInfo->tm_year + 1900,	// 从1900开始
		timeInfo->tm_mon + 1, // 原区间[0, 11]
		timeInfo->tm_mday,
		timeInfo->tm_hour,
		timeInfo->tm_min,
		timeInfo->tm_sec);

	return buf;
}



/////////////////////////////////////////////////////////////

// 每个连接的用户数据（可扩展）
struct PerClientData {
    // 可以在这里保存客户端特定状态
};

VRBridge::VRBridge(int port = 8080)
    : m_port(port), m_context(nullptr), m_running(false) {
}

VRBridge::~VRBridge() {
    stop();
}

/**
*  [1]使用
*/
bool VRBridge::start() {
    if (m_running) return true;

    // 设置日志级别
    lws_set_log_level(LLL_NOTICE | LLL_ERR | LLL_WARN, NULL);

    // 协议定义
    static struct lws_protocols protocols[] = {
        {
            "vr-bridge-protocol",    // 协议名称，需与客户端匹配
            wsCallback,              // 回调函数
            sizeof(PerClientData),   // 用户数据大小
            65536,                   // 接收缓冲区，设置大一些以防大JSON
            0, NULL, 0
        },
        { NULL, NULL, 0, 0 } // 结束标记
    };

    // 上下文配置
    struct lws_context_creation_info info;
    memset(&info, 0, sizeof(info));
    info.port = m_port;
    info.protocols = protocols;
    info.gid = -1;
    info.uid = -1;
    info.user = this; // 将 this 指针传给 context，以便回调中使用
    info.options = LWS_SERVER_OPTION_VALIDATE_UTF8;

    m_context = lws_create_context(&info);
    if (!m_context) {
        std::cerr << "Failed to create libwebsockets context" << std::endl;
        return false;
    }

    m_running = true;
    m_serviceThread = std::thread(&VRBridge::runService, this);

    std::cout << "VRBridge started on port " << m_port << std::endl;
    return true;
}

void VRBridge::stop() {
    if (!m_running) return;
    m_running = false;

    if (m_context) {
        // 通知线程退出
        lws_cancel_service(m_context);
    }

    if (m_serviceThread.joinable()) {
        m_serviceThread.join();
    }

    if (m_context) {
        lws_context_destroy(m_context);
        m_context = nullptr;
    }
}

/*
* [1]中 辅助函数
*/
// 后台线程：驱动 libwebsockets 事件循环
void VRBridge::runService() {
    while (m_running) {
        // lws_service 处理网络事件，timeout 为毫秒
        // 即使没有连接，也会定时返回以检查 m_running 状态
        lws_service(m_context, 50);

        // 使用 C++11 标准库休眠，替代 usleep，跨平台通用
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

/**
*  [3]使用
*/
// 发送数据接口：将结构体序列化为 JSON 并广播
void VRBridge::sendDetectionData(const FrameData& data) {
    try {
        // 使用 nlohmann/json 自动序列化
        nlohmann::json j = data;
        broadcastMessage(j.dump());
    }
    catch (const std::exception& e) {
        std::cerr << "JSON serialization error: " << e.what() << std::endl;
    }
}

/** [3] 中辅助函数
*   最终是使用[中转]完成发送操作
*/
// 广播消息：放入缓冲区并触发所有客户端写回调
void VRBridge::broadcastMessage(std::string&& message) {
    {
        // 更新最新数据帧（覆盖旧数据，保证实时性）
        std::lock_guard<std::mutex> lock(m_payloadMutex);
        m_latestPayload = std::move(message);
    }

    // 触发所有客户端的可写回调
    std::lock_guard<std::mutex> lock(m_clientMutex);
    for (auto wsi : m_clients) {
        // 这里的作用是让 libwebsockets 在下一帧调用 LWS_CALLBACK_SERVER_WRITEABLE
        lws_callback_on_writable(wsi);
    }
}

/**
*  [2]使用
*/
// 静态回调转发
int VRBridge::wsCallback(struct lws* wsi, enum lws_callback_reasons reason,
    void* user, void* in, size_t len) {

    struct lws_context* context = lws_get_context(wsi);
    VRBridge* self = static_cast<VRBridge*>(lws_context_user(context));

    if (self) {
        return self->handleCallback(wsi, reason, user, in, len);
    }
    return 0;
}

/*
* [中转： 2、] 整体(最终)操作 辅助函数
*/
// 实际回调处理
int VRBridge::handleCallback(struct lws* wsi, enum lws_callback_reasons reason,
    void* user, void* in, size_t len) {

    switch (reason) {
        case LWS_CALLBACK_ESTABLISHED: {
            // 新客户端连接
            std::lock_guard<std::mutex> lock(m_clientMutex);
            m_clients.push_back(wsi);
            lwsl_notice("Client connected: %p (total: %zu)\n", wsi, m_clients.size());
            break;
        }

        case LWS_CALLBACK_CLOSED: {
            // 客户端断开
            std::lock_guard<std::mutex> lock(m_clientMutex);
            auto it = std::remove(m_clients.begin(), m_clients.end(), wsi);
            m_clients.erase(it, m_clients.end());
            lwsl_notice("Client disconnected: %p (total: %zu)\n", wsi, m_clients.size());
            break;
        }

        case LWS_CALLBACK_SERVER_WRITEABLE: {
            // 可写事件：发送缓冲区中的最新数据
            std::string payloadToSend;
            {
                // 加锁拷贝数据，尽量减少锁持有时间
                std::lock_guard<std::mutex> lock(m_payloadMutex);
                payloadToSend = m_latestPayload;
            }

            if (!payloadToSend.empty()) {
                // libwebsockets 要求在缓冲区前预留 LWS_PRE 字节
                std::vector<unsigned char> buf(LWS_PRE + payloadToSend.size());
                memcpy(&buf[LWS_PRE], payloadToSend.data(), payloadToSend.size());

                // 发送文本帧
                int ret = lws_write(wsi, &buf[LWS_PRE], payloadToSend.size(), LWS_WRITE_TEXT);
                if (ret < 0) {
                    lwsl_err("Write error, closing connection\n");
                    return -1; // 返回 -1 会导致连接关闭
                }
            }
            break;
        }

        case LWS_CALLBACK_RECEIVE: {
            // 如果需要接收客户端指令，在此处理
            // lwsl_notice("Received: %s\n", (char*)in);
            break;
        }

        default:
            break;
    }
    return 0;
}