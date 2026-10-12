大致情况：
DealImg — TAGC 低照度图像增强（手写 float* 行指针逐像素遍历 + 自适应 gamma）
YoloDetect — ONNX Runtime 1.20.1 跑 YOLO11n，前后处理全部手写，没有用任何推理框架的封装
VRBridge — 基于 libwebsockets 的 C++ WebSocket 服务，把检测结果序列化成 JSON 广播给 VR 前端

主要流程：
图片 → 视频 → YOLO → 网络
