已实现整个程序中 图片和视频 OpenCV先(预)处理，之后Yolo对处理过的数据进行识别的功能；  

部分模块：  
DealImg — TAGC 低照度图像增强（手写 float* 行指针逐像素遍历 + 自适应 gamma）  
YoloDetect — ONNX Runtime 1.20.1 跑 YOLO11n，前后处理全部手写，没有用任何推理框架的封装  
VRBridge — 基于 libwebsockets 的 C++ WebSocket 服务，把检测结果序列化成 JSON 广播给 VR 前端  

主要流程：  
图片 → 视频 → YOLO → 网络   

待更新功能：  
[1] 初级版  
通过websocket通讯，实现利用前端(vue+three.js)实时显示/模拟 出后端的识别和处理的数据(类似仿真)  
[2] 进阶版  
同样通过websocket通讯，但将前端技术换为，使用更精细的UE  
