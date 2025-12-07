import cv2
import numpy as np
import time
from datetime import datetime
import os
import threading
import requests
from requests.auth import HTTPBasicAuth
import pyrealsense2 as rs
import platform

# 导入配置文件
try:
    from camera_config import *
except ImportError:
    print("警告: 无法导入camera_config.py，使用默认配置")
    # 默认配置
    IP_CAMERA_BASE = "192.168.1."
    IP_START = 11
    IP_END = 17
    CAMERA_USERNAME = "admin"
    CAMERA_PASSWORD = "admin"
    CAMERA_TYPE = "generic_rtsp"
    STREAM_URLS = {
        "generic_rtsp": "rtsp://{username}:{password}@{ip}:554/stream1"
    }
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    # DISPLAY_WIDTH = 2560
    # DISPLAY_HEIGHT = 1440
    SAVE_WIDTH = 1280
    SAVE_HEIGHT = 720
    FPS = 15
    CONNECTION_TIMEOUT = 10
    BUFFER_SIZE = 1
    OUTPUT_DIR = "ip_camera_photos"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# RealSense 相机配置（可选，从配置文件读取）
try:
    _ = USE_REALSENSE
except NameError:
    USE_REALSENSE = False

# 硬解码配置
try:
    USE_HARDWARE_DECODE
except NameError:
    USE_HARDWARE_DECODE = True

try:
    SYNC_THRESHOLD
except NameError:
    SYNC_THRESHOLD = 0.05

# USE_REALSENSE = True

try:
    _ = COLOR_WIDTH
except NameError:
    COLOR_WIDTH, COLOR_HEIGHT = 1280, 720

try:
    _ = DEPTH_WIDTH
except NameError:
    DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720

try:
    _ = MAX_DEPTH_DISPLAY
except NameError:
    MAX_DEPTH_DISPLAY = 3000

DEPTH_COLORMAP = cv2.COLORMAP_JET

# 全局变量
capture_flag = False
camera_locks = {}

class CameraThread(threading.Thread):
    """独立的相机拉流线程"""
    
    def __init__(self, ip, url, use_hardware_decode=False):
        super().__init__(daemon=True)
        self.ip = ip
        self.url = url
        self.use_hardware_decode = use_hardware_decode
        self.latest_frame = None
        self.latest_timestamp = None
        self.frame_lock = threading.Lock()  # 保护latest_frame的锁
        self.running = True
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.connection_lost = False
        self.retry_count = 0
        self.max_retries = 3
        
    def create_capture_pipeline(self):
        """创建VideoCapture对象，支持硬解码"""
        if self.use_hardware_decode:
            # 尝试使用GStreamer硬解码pipeline
            pipeline = self._create_gstreamer_pipeline()
            if pipeline:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    print(f"  {self.ip}: Using GStreamer hardware decode")
                    return cap
                else:
                    print(f"  {self.ip}: GStreamer hardware decode failed, falling back to software")
        
        # 回退到软解码
        cap = cv2.VideoCapture(self.url)
        if cap.isOpened():
            print(f"  {self.ip}: Using software decode")
        return cap
    
    def _create_gstreamer_pipeline(self):
        """创建GStreamer硬解码pipeline"""
        try:
            # 检测NVIDIA GPU
            if self._has_nvidia_gpu():
                # NVIDIA硬解码pipeline
                pipeline = (
                    f"rtspsrc location={self.url} latency=100 ! "
                    f"rtph264depay ! h264parse ! nvh264dec ! "
                    f"videoconvert ! video/x-raw,format=BGR ! appsink"
                )
            else:
                # Intel/AMD硬解码pipeline  
                pipeline = (
                    f"rtspsrc location={self.url} latency=100 ! "
                    f"rtph264depay ! h264parse ! avdec_h264 ! "
                    f"videoconvert ! video/x-raw,format=BGR ! appsink"
                )
            return pipeline
        except Exception as e:
            print(f"  {self.ip}: Failed to create GStreamer pipeline: {e}")
            return None
    
    def _has_nvidia_gpu(self):
        """检测是否有NVIDIA GPU"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def run(self):
        """线程主循环"""
        while self.running:
            cap = None
            try:
                # 创建VideoCapture
                cap = self.create_capture_pipeline()
                if not cap or not cap.isOpened():
                    raise Exception("Cannot open camera stream")
                
                # 配置相机参数
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, FPS)
                
                self.connection_lost = False
                self.retry_count = 0
                print(f"  {self.ip}: Stream connected")
                
                # 主拉流循环
                while self.running:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"  {self.ip}: Lost frame, retrying...")
                        break
                    
                    # 更新帧数据（使用锁保护）
                    current_time = time.time()
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.latest_timestamp = current_time
                    
                    # FPS统计
                    self.fps_counter += 1
                    if current_time - self.last_fps_time >= 1.0:
                        self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                        self.fps_counter = 0
                        self.last_fps_time = current_time
                    
                    time.sleep(0.001)  # 微小延迟减少CPU使用
                        
            except Exception as e:
                self.connection_lost = True
                self.retry_count += 1
                print(f"  {self.ip}: Connection error ({self.retry_count}/{self.max_retries}): {e}")
                
                if self.retry_count >= self.max_retries:
                    print(f"  {self.ip}: Max retries reached, stopping thread")
                    break
                
                # 重连延迟
                time.sleep(2.0 * self.retry_count)
            
            finally:
                if cap:
                    cap.release()
    
    def get_latest_frame(self):
        """获取最新帧（线程安全，返回copy）"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, None
    
    def stop(self):
        """停止线程"""
        self.running = False

class FrameSync:
    """帧同步管理器"""
    
    def __init__(self, sync_threshold=0.05):
        self.sync_threshold = sync_threshold  # 50ms同步阈值
        
    def get_synchronized_frames(self, camera_threads):
        """获取同步的帧"""
        frames = {}
        timestamps = {}
        
        # 收集所有相机的最新帧
        for thread in camera_threads:
            if thread.connection_lost:
                continue
                
            frame, timestamp = thread.get_latest_frame()
            if frame is not None and timestamp is not None:
                frames[thread.ip] = frame
                timestamps[thread.ip] = timestamp
        
        if not frames:
            return {}, {}
        
        # 找到参考时间戳（最新的）
        reference_time = max(timestamps.values())
        
        # 过滤掉时间差过大的帧
        synchronized_frames = {}
        synchronized_timestamps = {}
        
        for ip in frames.keys():
            time_diff = abs(timestamps[ip] - reference_time)
            if time_diff <= self.sync_threshold:
                synchronized_frames[ip] = frames[ip]
                synchronized_timestamps[ip] = timestamps[ip]
        
        return synchronized_frames, synchronized_timestamps

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数：检测左键点击"""
    global capture_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_flag = True

def generate_camera_url(ip):
    """根据IP地址生成相机URL"""
    url_template = STREAM_URLS[CAMERA_TYPE]
    return url_template.format(
        username=CAMERA_USERNAME,
        password=CAMERA_PASSWORD,
        ip=ip
    )

def test_camera_connection(ip):
    """测试相机连接"""
    url = generate_camera_url(ip)
    print(f"  Testing camera {ip}: {url.replace(CAMERA_PASSWORD, '***')}")
    
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return False, "Failed to open stream"
        
        # 设置超时
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 尝试读取一帧
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            return True, f"OK ({frame.shape[1]}x{frame.shape[0]})"
        else:
            return False, "No frame received"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def open_ip_camera(ip):
    """打开IP相机并配置"""
    url = generate_camera_url(ip)
    
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {ip}")
        
        # 设置缓冲区大小以减少延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # 创建线程锁
        camera_locks[ip] = threading.Lock()
        
        print(f"  IP Camera {ip}: Connected")
        return cap
        
    except Exception as e:
        print(f"  IP Camera {ip}: Failed - {e}")
        return None

def capture_frame_safe(camera, ip):
    """安全地从相机获取帧"""
    try:
        with camera_locks[ip]:
            ret, frame = camera.read()
            if ret and frame is not None:
                return frame
    except Exception as e:
        print(f"Error reading from camera {ip}: {e}")
    return None

def resize_frame(frame, width, height):
    """调整帧大小"""
    if frame is None:
        return None
    return cv2.resize(frame, (width, height))

def setup_realsense():
    """配置并启动 RealSense 相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        serial_number = device.get_info(rs.camera_info.serial_number)
        
        print(f"  RealSense: {device_name} (SN: {serial_number})")
        
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        return pipeline, align
        
    except Exception as e:
        print(f"  Warning: Failed to start RealSense camera: {e}")
        return None, None

def colorize_depth(depth_image):
    """将深度图像转换为彩色图像"""
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=255.0/MAX_DEPTH_DISPLAY), 
        DEPTH_COLORMAP
    )
    return depth_colormap

def main():
    global capture_flag
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # 生成IP列表
    # ip_list = [f"{IP_CAMERA_BASE}{i}" for i in range(IP_START, IP_END + 1)]
    ip_list = [f"{IP_CAMERA_BASE}{i}" for i in range(IP_START, IP_END + 1, 2)]
    # ip_list = ['192.168.1.16']
    print(ip_list)
    print(f"\nMulti-threaded IP Camera Configuration:")
    print(f"  Camera Type: {CAMERA_TYPE}")
    print(f"  IP Range: {ip_list[0]} - {ip_list[-1]} ({len(ip_list)} cameras)")
    print(f"  Username: {CAMERA_USERNAME}")
    print(f"  Threading: Enabled")
    print(f"  Hardware Decode: Auto-detect")
    
    # 测试所有相机连接
    print(f"\nTesting camera connections...")
    available_cameras = {}
    
    for ip in ip_list:
        success, message = test_camera_connection(ip)
        if success:
            print(f"  {ip}: ✓ {message}")
            available_cameras[ip] = None  # 将在后面打开
        else:
            print(f"  {ip}: ✗ {message}")
    
    if not available_cameras:
        print("\n❌ No cameras are available!")
        return
    
    print(f"\n✓ Found {len(available_cameras)} available camera(s)")
    
    # 打开 RealSense 相机（可选）
    realsense_pipeline = None
    realsense_align = None
    if USE_REALSENSE:
        print("\nInitializing RealSense camera...")
        realsense_pipeline, realsense_align = setup_realsense()
        if realsense_pipeline is None:
            print("Warning: RealSense camera not available, continuing with IP cameras only.")
            
    # 创建多线程相机系统
    print(f"\nInitializing multi-threaded camera system...")
    camera_threads = []
    print(f"  Hardware decode: {'Enabled' if USE_HARDWARE_DECODE else 'Disabled'}")
    
    for ip in available_cameras.keys():
        url = generate_camera_url(ip)
        thread = CameraThread(ip, url, USE_HARDWARE_DECODE)
        camera_threads.append(thread)
        thread.start()
    
    # 等待线程启动
    time.sleep(2.0)
    
    # 检查活动线程
    active_threads = [t for t in camera_threads if not t.connection_lost]
    if not active_threads:
        print("❌ Failed to start any camera threads!")
        return
    
    print(f"\n✓ Successfully started {len(active_threads)} camera thread(s)")
    
    
    # 创建帧同步器
    frame_sync = FrameSync(sync_threshold=SYNC_THRESHOLD)
    
    total_cameras = len(active_threads) + (1 if realsense_pipeline else 0)
    print(f"\n✓ Total cameras: {total_cameras} ({len(active_threads)} IP + {1 if realsense_pipeline else 0} RealSense)")
    
    # 创建显示窗口
    for thread in active_threads:
        window_name = f"IP Cam {thread.ip}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        cv2.setMouseCallback(window_name, mouse_callback)
    
    if realsense_pipeline:
        cv2.namedWindow("RealSense Color")
        cv2.setMouseCallback("RealSense Color", mouse_callback)
        cv2.namedWindow("RealSense Depth")
        cv2.setMouseCallback("RealSense Depth", mouse_callback)
    
    photo_count = 0
    print(f"\n{'='*60}")
    print(f"Multi-threaded IP Cameras Capture Ready ({total_cameras} cameras)")
    print(f"  - IP cameras: {len(active_threads)}")
    for i, thread in enumerate(active_threads):
        print(f"    Camera {i+1}: {thread.ip}")
    print(f"  - RealSense: {'Yes' if realsense_pipeline else 'No'}")
    print(f"  - Sync threshold: {frame_sync.sync_threshold*1000:.0f}ms")
    print("Click LEFT MOUSE BUTTON on any window to capture all photos")
    print("Press 'q' to quit")
    print(f"{'='*60}\n")
    
    try:
        while True:
            # 获取同步的帧
            frames, timestamps = frame_sync.get_synchronized_frames(active_threads)

            if not frames:
                # 检查是否有活动线程
                active_count = sum(1 for t in active_threads if not t.connection_lost)
                if active_count == 0:
                    print("All camera threads lost connection!")
                    break
                time.sleep(0.01)
                continue
            
            # 读取 RealSense 帧
            realsense_color = None
            realsense_depth = None
            realsense_depth_viz = None
            
            if realsense_pipeline:
                try:
                    frames_rs = realsense_pipeline.wait_for_frames(timeout_ms=100)
                    aligned_frames = realsense_align.process(frames_rs)
                    
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if color_frame and depth_frame:
                        realsense_color = np.asanyarray(color_frame.get_data())
                        realsense_depth = np.asanyarray(depth_frame.get_data())
                        realsense_depth_viz = colorize_depth(realsense_depth)
                except Exception as e:
                    pass  # 继续运行，即使RealSense暂时失败
            
            if not frames and realsense_color is None:
                continue
            
            # 显示所有相机画面
            for ip, frame in frames.items():
                # 调整显示大小
                display_frame = resize_frame(frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                if display_frame is not None:
                    # 找到对应的线程获取FPS
                    fps = 0
                    for thread in active_threads:
                        if thread.ip == ip:
                            fps = thread.current_fps
                            break
                    
                    # 添加信息文本
                    sync_count = len(frames)
                    info_text = f"{ip} | Photos: {photo_count} | FPS: {fps:.1f} | Sync: {sync_count}/{len(active_threads)}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.putText(display_frame, "Click to capture", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    cv2.imshow(f"IP Cam {ip}", display_frame)
            
            # 显示 RealSense 画面
            if realsense_color is not None:
                info_text = f"RealSense | Photos: {photo_count} | Click to capture"
                cv2.putText(realsense_color, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('RealSense Color', realsense_color)
            
            if realsense_depth_viz is not None:
                cv2.imshow('RealSense Depth', realsense_depth_viz)
            
            # 检查是否点击了鼠标左键
            if capture_flag:
                timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
                
                # 保存所有相机的照片
                saved_files = []
                
                # 保存IP相机照片
                for ip, frame in frames.items():
                    # 调整保存分辨率
                    save_frame = resize_frame(frame, SAVE_WIDTH, SAVE_HEIGHT)
                    if save_frame is not None:
                        filename = os.path.join(OUTPUT_DIR, f"ipcam_{ip}_{timestamp}.jpg")
                        success = cv2.imwrite(filename, save_frame)
                        if success:
                            saved_files.append(filename)
                            print(f"  ✓ {filename}")
                        else:
                            print(f"  ✗ Failed to save {filename}")
                
                # 保存 RealSense 照片
                if realsense_color is not None and realsense_depth is not None:
                    color_filename = os.path.join(OUTPUT_DIR, f"realsense_color_{timestamp}.jpg")
                    depth_filename = os.path.join(OUTPUT_DIR, f"realsense_depth_{timestamp}.png")
                    depth_viz_filename = os.path.join(OUTPUT_DIR, f"realsense_depth_viz_{timestamp}.jpg")
                    
                    cv2.imwrite(color_filename, realsense_color)
                    cv2.imwrite(depth_filename, realsense_depth)
                    cv2.imwrite(depth_viz_filename, realsense_depth_viz)
                    
                    saved_files.extend([color_filename, depth_filename, depth_viz_filename])
                    print(f"  ✓ {color_filename}")
                    print(f"  ✓ {depth_filename}")
                    print(f"  ✓ {depth_viz_filename}")
                
                photo_count += 1
                total_cameras = len(frames) + (1 if realsense_color is not None else 0)
                print(f"\n[{photo_count}] Captured {len(saved_files)} files from {total_cameras} cameras: {timestamp}")
                
                capture_flag = False
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # 释放所有资源
        print("\nCleaning up...")
        
        # 停止所有相机线程
        for thread in camera_threads:
            try:
                thread.stop()
                print(f"  Stopping camera thread {thread.ip}")
            except:
                pass
        
        # 等待线程结束
        for thread in camera_threads:
            try:
                thread.join(timeout=2.0)
                print(f"  Camera thread {thread.ip} stopped")
            except:
                pass
        
        if realsense_pipeline:
            realsense_pipeline.stop()
            print(f"  Released RealSense camera")
        
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"Done! Total photo sets captured: {photo_count}")
        print(f"Photos saved in: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Multi-threading performance summary:")
        for thread in camera_threads:
            if not thread.connection_lost:
                print(f"  {thread.ip}: Final FPS: {thread.current_fps:.1f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    print("IP Camera Capture System")
    print("=" * 40)
    main()
