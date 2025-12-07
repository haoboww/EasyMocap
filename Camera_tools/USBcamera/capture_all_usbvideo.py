import cv2
import numpy as np
import pyrealsense2 as rs
import time
from datetime import datetime
import os

# ==================== 相机配置 ====================
# 普通相机索引列表
CAMERA_INDICES = [0, 3, 4]  # 3台普通相机的索引

# RealSense 相机配置
USE_REALSENSE = True                      # 是否使用RealSense相机
COLOR_WIDTH, COLOR_HEIGHT = 1280, 720     # RealSense彩色图像分辨率
DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720     # RealSense深度图像分辨率

# 普通相机参数
WIDTH, HEIGHT = 1280, 720
FPS = 30

# 录制参数
OUTPUT_FORMAT = "avi"         # 输出格式：avi, mp4
CODEC = "XVID"                # 编码器：XVID, MJPG, mp4v
OUTPUT_DIR = "all_videos"     # 输出目录

# RealSense深度保存方式
SAVE_DEPTH_AS_PNG = True      # True: 保存为16位PNG序列(推荐), False: 保存为avi视频

# 对焦参数（定焦镜头通常不需要）
ENABLE_FOCUS_CONTROL = False              # 定焦镜头设为False
AUTO_FOCUS = False
FOCUS_VALUE = 45

# 深度图可视化参数
DEPTH_COLORMAP = cv2.COLORMAP_JET
MAX_DEPTH_DISPLAY = 3000                  # 显示的最大深度值(mm)

# 全局变量
recording_started = False  # 标记是否开始录制
stop_recording = False     # 标记是否需要停止录制

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数：检测左键点击启动/停止录制"""
    global recording_started, stop_recording
    if event == cv2.EVENT_LBUTTONDOWN:
        if not recording_started:
            recording_started = True
            print("\n>>> Mouse clicked - Starting recording NOW! <<<")
        else:
            stop_recording = True
            print("\n>>> Mouse clicked - Stopping recording... <<<")

def open_normal_cam(index):
    """打开并配置普通相机"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    
    if ENABLE_FOCUS_CONTROL:
        if not AUTO_FOCUS:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            time.sleep(0.1)
            cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VALUE)
            print(f"  Normal Camera {index}: Manual focus = {FOCUS_VALUE}")
        else:
            print(f"  Normal Camera {index}: Auto-focus enabled")
    else:
        print(f"  Normal Camera {index}: Fixed lens mode")
    
    return cap

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
        #  dddd
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print(f"Depth Scale: {depth_scale}")  

        # 创建对齐对象
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
    global recording_started, stop_recording
    
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # 时间戳将在开始录制时生成
    timestamp = None
    
    # 打开所有普通相机
    normal_cameras = {}
    print(f"\nOpening {len(CAMERA_INDICES)} normal camera(s)...")
    for i, cam_index in enumerate(CAMERA_INDICES):
        try:
            normal_cameras[i] = open_normal_cam(cam_index)
            print(f"Normal Camera {i} (index {cam_index}): OK")
        except Exception as e:
            print(f"Failed to open camera {i} (index {cam_index}): {e}")
            for cam in normal_cameras.values():
                cam.release()
            return
    
    # 打开 RealSense 相机
    realsense_pipeline = None
    realsense_align = None
    if USE_REALSENSE:
        print("\nInitializing RealSense camera...")
        realsense_pipeline, realsense_align = setup_realsense()
        if realsense_pipeline is None:
            print("Warning: RealSense camera not available, continuing with normal cameras only.")
    
    total_cameras = len(normal_cameras) + (1 if realsense_pipeline else 0)
    print(f"\nAll cameras opened successfully! Total: {total_cameras} camera(s)")
    
    # 视频写入器将在开始录制时创建
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    normal_writers = {}
    realsense_color_writer = None
    realsense_depth_writer = None
    realsense_depth_dir = None
    
    # 创建窗口并设置鼠标回调
    for i in range(len(normal_cameras)):
        window_name = f"Normal Cam {i}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
    
    if realsense_pipeline:
        cv2.namedWindow("RealSense Color")
        cv2.setMouseCallback("RealSense Color", mouse_callback)
        cv2.namedWindow("RealSense Depth")
        cv2.setMouseCallback("RealSense Depth", mouse_callback)
    
    print(f"\n{'='*60}")
    print(f"Preview Mode ({total_cameras} cameras)")
    print(f"  - Normal cameras: {len(normal_cameras)}")
    print(f"  - RealSense: {'Yes' if realsense_pipeline else 'No'}")
    print(f"  - Resolution: {WIDTH}x{HEIGHT} @ {FPS} FPS")
    print(f"  - Codec: {CODEC} | Format: {OUTPUT_FORMAT}")
    if realsense_pipeline:
        depth_mode = "16-bit PNG sequence" if SAVE_DEPTH_AS_PNG else "avi video (lossy)"
        print(f"  - Depth save mode: {depth_mode}")
    if not ENABLE_FOCUS_CONTROL:
        print("  - Focus: Fixed lens mode")
    print("")
    print(">>> Press SPACE or Click MOUSE to START recording <<<")
    print(">>> Press 'q' to quit <<<")
    print(f"{'='*60}\n")
    
    start_time = None
    frame_count = 0

    try:
        while True:
            # 读取所有普通相机的帧
            normal_frames = {}
            all_frames_ok = True
            
            for i, camera in normal_cameras.items():
                ret, frame = camera.read()
                if ret:
                    normal_frames[i] = frame
                else:
                    print(f"Failed to grab frame from normal camera {i}")
                    all_frames_ok = False
                    break
            
            if not all_frames_ok:
                if recording_started:
                    print("Camera read error, stopping recording...")
                break
            
            # 读取 RealSense 帧
            realsense_color = None
            realsense_depth = None
            realsense_depth_viz = None
            
            if realsense_pipeline:
                try:
                    frames = realsense_pipeline.wait_for_frames(timeout_ms=100)
                    aligned_frames = realsense_align.process(frames)
                    
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if color_frame and depth_frame:
                        realsense_color = np.asanyarray(color_frame.get_data())
                        realsense_depth = np.asanyarray(depth_frame.get_data())
                        realsense_depth_viz = colorize_depth(realsense_depth)
                except Exception as e:
                    pass  # 继续运行，即使RealSense暂时失败
            
            # 检查是否开始录制
            if recording_started and start_time is None:
                # 首次开始录制，创建视频写入器
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                start_time = time.time()
                
                # 创建普通相机的视频写入器
                for i, cam_index in enumerate(CAMERA_INDICES):
                    output_filename = os.path.join(
                        OUTPUT_DIR, 
                        f"normal_cam{i}_idx{cam_index}_{timestamp}.{OUTPUT_FORMAT}"
                    )
                    normal_writers[i] = cv2.VideoWriter(
                        output_filename, 
                        fourcc, 
                        FPS, 
                        (WIDTH, HEIGHT)
                    )
                    print(f"Recording normal cam {i} to: {output_filename}")
                
                # 创建 RealSense 的视频写入器
                if realsense_pipeline:
                    color_filename = os.path.join(OUTPUT_DIR, f"realsense_color_{timestamp}.{OUTPUT_FORMAT}")
                    realsense_color_writer = cv2.VideoWriter(
                        color_filename, fourcc, FPS, (COLOR_WIDTH, COLOR_HEIGHT)
                    )
                    print(f"Recording RealSense color to: {color_filename}")
                    
                    if SAVE_DEPTH_AS_PNG:
                        realsense_depth_dir = os.path.join(OUTPUT_DIR, f"realsense_depth_{timestamp}")
                        os.makedirs(realsense_depth_dir, exist_ok=True)
                        print(f"Recording RealSense depth (16-bit PNG) to: {realsense_depth_dir}/")
                    else:
                        depth_filename = os.path.join(OUTPUT_DIR, f"realsense_depth_{timestamp}.{OUTPUT_FORMAT}")
                        realsense_depth_writer = cv2.VideoWriter(
                            depth_filename, fourcc, FPS, (DEPTH_WIDTH, DEPTH_HEIGHT)
                        )
                        print(f"Recording RealSense depth to: {depth_filename}")
                
                print(f"\n{'='*60}")
                print(f">>> RECORDING IN PROGRESS <<<")
                print(f"Click MOUSE or press 'q' to stop")
                print(f"{'='*60}\n")
            
            # 写入视频文件（仅在录制状态）
            if recording_started and start_time is not None:
                # 写入所有普通相机的视频
                for i, frame in normal_frames.items():
                    if i in normal_writers:
                        normal_writers[i].write(frame)
                
                # 写入 RealSense 数据
                if realsense_color is not None and realsense_color_writer:
                    realsense_color_writer.write(realsense_color)
                
                if realsense_depth is not None:
                    if SAVE_DEPTH_AS_PNG and realsense_depth_dir:
                        depth_filename = os.path.join(realsense_depth_dir, f"{frame_count:06d}.png")
                        cv2.imwrite(depth_filename, realsense_depth)
                    elif realsense_depth_writer and realsense_depth_viz is not None:
                        realsense_depth_writer.write(realsense_depth_viz)
                
                frame_count += 1
                
                # 每100帧显示一次进度
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Recording... Frame: {frame_count}, Time: {elapsed_time:.1f}s, FPS: {frame_count/elapsed_time:.2f}")
            
            # 显示所有普通相机画面
            for i, frame in normal_frames.items():
                frame_display = frame.copy()
                if recording_started and start_time is not None:
                    elapsed_time = time.time() - start_time
                    info_text = f"Cam {i} | REC | Frame: {frame_count} | Time: {elapsed_time:.1f}s"
                    color = (0, 0, 255)  # 红色表示录制中
                else:
                    info_text = f"Cam {i} | PREVIEW - Press SPACE to START"
                    color = (0, 255, 255)  # 黄色表示预览
                cv2.putText(frame_display, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow(f"Normal Cam {i}", frame_display)
            
            # 显示 RealSense 画面
            if realsense_color is not None:
                color_display = realsense_color.copy()
                if recording_started and start_time is not None:
                    elapsed_time = time.time() - start_time
                    info_text = f"RealSense | REC | Frame: {frame_count} | Time: {elapsed_time:.1f}s"
                    color = (0, 0, 255)  # 红色表示录制中
                else:
                    info_text = f"RealSense | PREVIEW - Press SPACE to START"
                    color = (0, 255, 255)  # 黄色表示预览
                cv2.putText(color_display, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imshow('RealSense Color', color_display)
            
            if realsense_depth_viz is not None:
                cv2.imshow('RealSense Depth', realsense_depth_viz)
            
            # 检查是否点击鼠标停止录制
            if stop_recording:
                break
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not recording_started:
                # 按空格键启动录制
                recording_started = True
                print("\n>>> Space pressed - Starting recording NOW! <<<")
    
    finally:
        # 释放所有资源
        for camera in normal_cameras.values():
            camera.release()
        
        for writer in normal_writers.values():
            writer.release()
        
        if realsense_pipeline:
            realsense_pipeline.stop()
        
        if realsense_color_writer:
            realsense_color_writer.release()
        if realsense_depth_writer:
            realsense_depth_writer.release()
        
        cv2.destroyAllWindows()
        
        duration = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Recording completed!")
        print(f"Duration: {duration:.1f}s")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {frame_count/duration:.2f}")
        print(f"Videos saved in: {os.path.abspath(OUTPUT_DIR)}")
        if realsense_depth_dir:
            print(f"RealSense depth (16-bit PNG): {os.path.abspath(realsense_depth_dir)}/")
            print(f"  Total depth images: {frame_count}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()

