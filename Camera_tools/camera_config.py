# IP相机配置文件
# 根据你的实际相机型号和网络配置修改以下参数

# ==================== 网络配置 ====================
IP_CAMERA_BASE = "192.168.1."
IP_START = 11  # 起始IP
IP_END = 17    # 结束IP (共8台相机: 192.168.1.10 - 192.168.1.17)

# ==================== 认证配置 ====================
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = ""  # 请修改为你的相机密码

# ==================== 相机品牌配置 ====================
# 根据你的相机品牌选择对应的流地址格式
STREAM_URLS = {
    # 海康威视 (Hikvision)
    "hikvision": "rtsp://{username}:{password}@{ip}:554/Streaming/Channels/101",
    "hikvision_h264": "rtsp://{username}:{password}@{ip}:554/h264/ch1/main/av_stream",
    
    # 大华 (Dahua)
    "dahua": "rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "dahua_main": "rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=1",
    
    # 宇视 (Uniview)
    "uniview": "rtsp://{username}:{password}@{ip}:554/unicast/c1/s0/live",
    
    # TP-Link
    "tplink": "rtsp://{username}:{password}@{ip}:554/stream1",
    
    # 通用RTSP格式
    "generic_rtsp": "rtsp://{username}:{password}@{ip}:554/stream1",
    "generic_rtsp2": "rtsp://{username}:{password}@{ip}:554/live/0/MAIN",
    
    # MJPEG HTTP流 (适用于较老的相机)
    "mjpeg": "http://{username}:{password}@{ip}/mjpg/video.mjpg",
    "mjpeg2": "http://{username}:{password}@{ip}/videostream.cgi",
    
    # ONVIF标准格式
    "onvif": "rtsp://{username}:{password}@{ip}:554/onvif1"
}

# 当前使用的相机类型 (修改为你的相机品牌)
CAMERA_TYPE = "generic_rtsp"

# ==================== 图像参数 ====================
# 显示窗口大小
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# 保存图片的分辨率
# SAVE_WIDTH = 1280
# SAVE_HEIGHT = 720
SAVE_WIDTH = 1920
SAVE_HEIGHT = 1080
# 帧率
FPS = 15

# ==================== 连接参数 ====================
CONNECTION_TIMEOUT = 10    # 连接超时时间(秒)
FRAME_TIMEOUT = 1000      # 帧获取超时时间(毫秒)
BUFFER_SIZE = 1           # 缓冲区大小(减少延迟)

# ==================== 输出配置 ====================
OUTPUT_DIR = "ip_camera_photos"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# ==================== RealSense 相机配置（可选）====================
USE_REALSENSE = True                     # 是否使用RealSense相机
COLOR_WIDTH, COLOR_HEIGHT = 1280, 720      # RealSense彩色图像分辨率
DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720      # RealSense深度图像分辨率
SAVE_DEPTH_AS_PNG = True                   # True: 保存为16位PNG序列(推荐), False: 保存为avi视频
MAX_DEPTH_DISPLAY = 3000                   # 显示的最大深度值(mm)
# 注意: DEPTH_COLORMAP 在脚本中定义，无需在此配置

# ==================== 常用相机配置示例 ====================
"""
海康威视相机:
CAMERA_TYPE = "hikvision"
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "your_password"

大华相机:
CAMERA_TYPE = "dahua" 
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "your_password"

TP-Link相机:
CAMERA_TYPE = "tplink"
CAMERA_USERNAME = "admin" 
CAMERA_PASSWORD = "your_password"

如果以上都不work，尝试:
CAMERA_TYPE = "generic_rtsp2"
或者
CAMERA_TYPE = "onvif"
"""
