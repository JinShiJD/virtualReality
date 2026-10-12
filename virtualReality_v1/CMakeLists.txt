
cmake_minimum_required (VERSION 3.8)

project ("virtualReality_v1")
set(CMAKE_CXX_STANDARD 17)


set(OPENCVROOT_DIR D:/vs2019_files01/opencv/opencv/build/include)
set(OPENCV01_DIR ${OPENCVROOT_DIR}/opencv2)
set(OPENCV02_DIR ${OPENCVROOT_DIR}/opencv2/objdetect)

#三方库路径
#ONNXRuntime
set(ONNXRUNTIME_INCLUDE_DIR D:/vs2019_files02/ONNXRuntime/onnxruntime-win-x64-1.20.1/onnxruntime-win-x64-1.20.1)
#nlohmann/json
set(NLOHMANN_INCLUDE_DIR D:/vs2019_files02/virtualReality_v1/third_parties)
#libwebsockets
set(LIBWEBSOCKETS_INCLUDE_DIR D:/vcpkg/vcpkg-2025.06.13/installed/x64-windows)



#执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/out/build/x64-Debug)

#库文件路径
link_directories(D:/vs2019_files01/opencv/opencv/build/x64/vc16/lib)
#link_libraries(opencv_world4120d opencv_world4120)

link_directories(${ONNXRUNTIME_INCLUDE_DIR}/lib)
#link_libraries(onnxruntime onnxruntime_providers_shared)

#libwebsockets--Debug
link_directories(${LIBWEBSOCKETS_INCLUDE_DIR}/debug/lib)
link_directories(${LIBWEBSOCKETS_INCLUDE_DIR}/debug/bin)

#libwebsockets--Release
#link_directories(${LIBWEBSOCKETS_INCLUDE_DIR}/lib)
#link_directories(${LIBWEBSOCKETS_INCLUDE_DIR}/bin)

# 编译的源文件(.h 和 .cpp)
file(GLOB INCLUDE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

#三方库--头文件--#include
include_directories(${OPENCVROOT_DIR})
include_directories(${OPENCV01_DIR})
include_directories(${OPENCV02_DIR})

#onnxRuntime
include_directories(${ONNXRUNTIME_INCLUDE_DIR}/include)
#nlohmann/json
include_directories(${NLOHMANN_INCLUDE_DIR})
#libwebsockets
include_directories(${LIBWEBSOCKETS_INCLUDE_DIR}/include)


# 将源代码添加到此项目的可执行文件。
add_executable (virtualReality_v1 ${INCLUDE_LIST}  ${SRC_LIST})
#链接动态库 opencv_videoio_ffmpeg4120_64 optimized opencv_world4120
target_link_libraries(virtualReality_v1 
						opencv_world4120d 						
						onnxruntime
						onnxruntime_providers_shared
						websockets	
						libssl-3-x64
						libcrypto-3-x64						
						zlibd
						uv
						)

