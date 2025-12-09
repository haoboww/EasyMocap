#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
毫米波雷达数据采集工具 - 按帧保存为 bin 文件
"""
import argparse
import os
import threading
import time
from datetime import datetime
from typing import Optional

import serial
import usb.core
import usb.util


USB_VID = 0x04B4
USB_PID = 0x00F0
SERIAL_BAUDRATE = 2_000_000  # 2 Mbps
RESET_CMD = bytes.fromhex("41 54 2B 52 45 53 45 54 0A")   # AT+RESET
START_CMD = bytes.fromhex("41 54 2B 53 54 41 52 54 0A")   # AT+START
STOP_CMD = bytes.fromhex("41 54 2B 53 54 4F 50 0A")       # AT+STOP


def start_radar_bin(port: str):
    """初始化串口和 USB Bulk 端点。"""
    print(f"[INIT] 正在打开串口: {port} @ {SERIAL_BAUDRATE} bps")
    try:
        ser = serial.Serial(port, SERIAL_BAUDRATE, timeout=0.5)
    except serial.SerialException as e:
        raise RuntimeError(f"串口打开失败: {e}")
    
    print(f"[INIT] 串口已打开: {ser.name}")
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print(f"[INIT] 正在查找 USB 设备 (VID=0x{USB_VID:04X}, PID=0x{USB_PID:04X})...")
    dev = usb.core.find(idVendor=USB_VID, idProduct=USB_PID)
    if dev is None:
        # 列出所有 USB 设备帮助调试
        print("[INIT] 未找到目标设备，列出所有 USB 设备:")
        for d in usb.core.find(find_all=True):
            print(f"  - VID=0x{d.idVendor:04X}, PID=0x{d.idProduct:04X}")
        raise RuntimeError("未找到 FX3 设备 (VID=0x04B4, PID=0x00F0)")
    
    print(f"[INIT] 找到设备: {dev}")
    
    try:
        cfg = dev.get_active_configuration()
    except usb.core.USBError as e:
        print(f"[INIT] 无法获取配置，尝试设置配置... ({e})")
        dev.set_configuration()
        cfg = dev.get_active_configuration()
    
    print(f"[INIT] 当前配置: {cfg}")
    intf = cfg[(0, 0)]
    print(f"[INIT] 接口: {intf}")
    
    usb_ep_in = usb.util.find_descriptor(
        intf,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        and usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK,
    )
    if usb_ep_in is None:
        print("[INIT] 未找到 Bulk IN 端点，列出所有端点:")
        for ep in intf:
            print(f"  - Endpoint 0x{ep.bEndpointAddress:02X}, Type={ep.bmAttributes}")
        raise RuntimeError("未找到 Bulk IN 端点")
    
    print(f"[INIT] Bulk IN 端点: 0x{usb_ep_in.bEndpointAddress:02X}, MaxPacketSize={usb_ep_in.wMaxPacketSize}")
    return ser, usb_ep_in


def finish_radar_bin(ser: serial.Serial):
    """发送雷达复位指令并安全关闭串口。"""
    try:
        ser.write(RESET_CMD)
        time.sleep(0.5)
    finally:
        if ser.is_open:
            ser.close()
            print("串口已关闭。")


def _send_command(ser: serial.Serial, cmd: bytes, delay: float = 0.5) -> Optional[str]:
    """向雷达发送串口命令并读取回复。"""
    ser.write(cmd)
    time.sleep(delay)
    if ser.in_waiting > 0:
        response = ser.read(ser.in_waiting).decode(errors="ignore").strip()
        if response:
            print(f"[RADAR] Response: {response}")
        return response
    return None


def save_radar_bin(
    ser: serial.Serial,
    usb_ep_in,
    save_path: str,
    save_event: threading.Event,
    frame_size_bytes: int = 1_048_576,
    chunk_size: int = 4096,
    timeout_ms: int = 2000,
    max_frames: Optional[int] = None,
):
    """
    按帧保存雷达数据到 bin 文件。

    Parameters
    ----------
    ser : serial.Serial
        已经打开的串口对象。
    usb_ep_in : usb.core.Endpoint
        FX3 的 Bulk IN 端点。
    save_path : str
        输出目录，每帧一个 bin 文件。
    save_event : threading.Event
        当事件为 set() 时开始缓存并写文件，为 clear() 时仅丢弃数据。
    frame_size_bytes : int
        每帧应当包含的字节数，默认 1MB。
    chunk_size : int
        每次从 FX3 读取的字节数，默认 4096。
    timeout_ms : int
        USB 读取超时时间。
    max_frames : int | None
        若指定，则保存指定帧数后自动退出。
    """
    # 验证 chunk_size 是否为 MaxPacketSize 的倍数（优化性能）
    max_packet_size = usb_ep_in.wMaxPacketSize
    if chunk_size % max_packet_size != 0:
        print(f"[RADAR] 警告: chunk_size ({chunk_size}) 不是 MaxPacketSize ({max_packet_size}) 的倍数，可能影响性能")
    
    os.makedirs(save_path, exist_ok=True)
    print(f"[RADAR] 输出目录: {save_path}")

    print("[RADAR] 正在复位设备...")
    _send_command(ser, RESET_CMD)
    print("[RADAR] 正在启动数据采集...")
    _send_command(ser, START_CMD)
    print("[RADAR] 数据接收已开始 (AT+START 已发送)")
    print(f"[RADAR] 帧大小: {frame_size_bytes:,} 字节, 块大小: {chunk_size:,} 字节")
    print(f"[RADAR] MaxPacketSize: {max_packet_size} 字节")
    print(f"[RADAR] 保存状态: {'已启用' if save_event.is_set() else '等待触发'}")

    buffer = bytearray()
    frame_index = 0
    total_bytes_received = 0
    start_time = time.time()  # 记录开始时间用于计算平均速率
    last_report_time = time.time()
    last_bytes_count = 0  # 用于准确计算速率
    chunks_received = 0

    while True:
        if max_frames is not None and frame_index >= max_frames:
            print(f"[RADAR] 已保存 {frame_index} 帧，停止采集。")
            break

        try:
            chunk = usb_ep_in.read(chunk_size, timeout=timeout_ms)
            
            # 只有在成功读取到有效数据时才处理
            if chunk and len(chunk) > 0:
                chunks_received += 1
                total_bytes_received += len(chunk)
                
                # 每 100 个块报告一次接收状态
                if chunks_received % 100 == 0:
                    elapsed = time.time() - last_report_time
                    if elapsed > 0:
                        # 使用实际接收的字节数计算速率（更准确）
                        bytes_received = total_bytes_received - last_bytes_count
                        rate_mbps = (bytes_received * 8) / (elapsed * 1_000_000)
                        print(f"[RADAR] 接收中... 已收到 {chunks_received} 块 ({total_bytes_received:,} 字节), 速率: {rate_mbps:.2f} Mbps")
                    last_report_time = time.time()
                    last_bytes_count = total_bytes_received
                
                # 如果保存未启用，直接丢弃数据（不累积到 buffer）
                if not save_event.is_set():
                    continue
                
                # 累积数据到缓冲区
                buffer.extend(chunk)
                
                # 保存完整帧
                while len(buffer) >= frame_size_bytes:
                    frame_bytes = bytes(buffer[:frame_size_bytes])
                    del buffer[:frame_size_bytes]
                    
                    # 使用 Unix 时间戳（毫秒精度）
                    timestamp_ms = int(time.time() * 1000)
                    file_path = os.path.join(save_path, f"frame_{timestamp_ms}.bin")
                    
                    try:
                        with open(file_path, "wb") as f:
                            f.write(frame_bytes)
                        frame_index += 1
                        print(f"[RADAR] 帧 {frame_index} 已保存 -> {os.path.basename(file_path)} ({frame_size_bytes:,} 字节)")
                    except IOError as e:
                        print(f"[RADAR] 错误: 无法写入文件 {file_path}: {e}")
                        # 继续采集，不中断
                
        except usb.core.USBError as err:
            # errno 110 表示 timeout，可忽略继续读
            if getattr(err, "errno", None) in (None, 110):
                if chunks_received == 0:
                    print("[RADAR] 警告: USB 读取超时，未收到任何数据。检查设备是否正常工作。")
                continue
            print(f"[RADAR] USB 错误: {err}")
            raise
        except Exception as e:
            print(f"[RADAR] 未预期的错误: {e}")
            import traceback
            traceback.print_exc()
            # 发生严重错误时退出循环
            break
        
        time.sleep(0.001)
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    print(f"[RADAR] 总共接收: {total_bytes_received:,} 字节, {chunks_received} 块")
    if total_elapsed > 0 and total_bytes_received > 0:
        avg_rate = (total_bytes_received * 8) / (total_elapsed * 1_000_000)
        print(f"[RADAR] 总耗时: {total_elapsed:.2f} 秒, 平均速率: {avg_rate:.2f} Mbps")

    print("[RADAR] 发送停止命令...")
    _send_command(ser, STOP_CMD)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集毫米波雷达数据并以帧为单位保存为 bin")
    parser.add_argument("--port", required=True, help="串口端口，例如 COM4 或 /dev/ttyACM0")
    parser.add_argument("--output", required=True, help="保存 bin 文件的目录")
    parser.add_argument(
        "--frame-size",
        type=int,
        default=1_048_576,
        help="每帧数据大小（字节），默认 1,048,576",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="单次 USB 读取的字节数，默认 4,096",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2000,
        help="USB 读取超时时间（毫秒）",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="若指定，则采集到指定帧数后自动退出",
    )
    parser.add_argument(
        "--manual-start",
        action="store_true",
        help="启用手动开始保存，按回车后才写入 bin",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("毫米波雷达数据采集工具")
    print("=" * 60)
    print(f"串口: {args.port}")
    print(f"输出目录: {args.output}")
    print(f"帧大小: {args.frame_size:,} 字节")
    print(f"块大小: {args.chunk_size:,} 字节")
    print(f"超时: {args.timeout} ms")
    if args.max_frames:
        print(f"最大帧数: {args.max_frames}")
    print("=" * 60)
    
    save_event = threading.Event()
    
    # 如果不是手动模式，立即启用保存
    if not args.manual_start:
        save_event.set()
        print("[INFO] 自动保存模式已启用")
    else:
        print("[INFO] 手动模式: 等待用户输入...")
    
    ser = None
    try:
        print("\n[STEP 1/3] 初始化设备...")
        ser, usb_ep_in = start_radar_bin(args.port)
        print("[STEP 1/3] ✓ 设备初始化完成\n")
        
        # 验证参数合理性
        if args.frame_size <= 0:
            raise ValueError(f"帧大小必须大于 0，当前值: {args.frame_size}")
        if args.chunk_size <= 0:
            raise ValueError(f"块大小必须大于 0，当前值: {args.chunk_size}")
        if args.chunk_size > args.frame_size:
            print(f"[WARN] 警告: 块大小 ({args.chunk_size:,}) 大于帧大小 ({args.frame_size:,})，建议调整")
        
        if args.manual_start:
            print("[STEP 2/3] 等待用户确认...")
            input("按回车键开始保存帧数据...")
            save_event.set()
            print("[STEP 2/3] ✓ 开始保存\n")
        else:
            print("[STEP 2/3] 自动开始\n")
        
        print("[STEP 3/3] 开始采集数据...")
        print("提示: 按 Ctrl+C 可随时停止采集\n")
        
        save_radar_bin(
            ser,
            usb_ep_in,
            args.output,
            save_event,
            frame_size_bytes=args.frame_size,
            chunk_size=args.chunk_size,
            timeout_ms=args.timeout,
            max_frames=args.max_frames,
        )
    except KeyboardInterrupt:
        print("\n[INFO] 捕获到键盘中断 (Ctrl+C)，正在安全退出...")
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ser is not None:
            print("\n[CLEANUP] 正在关闭设备...")
            finish_radar_bin(ser)
        print("[CLEANUP] 采集已结束。")
        print("=" * 60)

if __name__ == "__main__":
    main()

