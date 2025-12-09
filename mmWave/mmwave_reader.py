#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mmwave_reader.py
----------------
A standalone Python script for parsing raw .bin ADC data from mmWave radar.
It extracts radar frames, reconstructs per-antenna ADC data,
visualizes signals, and saves results to .mat.

Compatible with TI IWR6843 / AWR1843 and similar devices.

Usage:
    python mmwave_reader.py --data_dir ./data --num_frames 50
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from datetime import datetime

# ==========================
# 雷达配置参数区域
# ==========================
CONFIG = {
    "numADCBits": 16,       # 每个ADC采样位数
    "numRx": 4,             # 接收天线数 (Rx)
    "numTx": 4,             # 发射天线数 (Tx) —— 若有 MIMO，则虚拟天线数 = numRx * numTx
    "numSamplesPerChirp": 512,  # 每个chirp的采样点数
    "numChirpsPerFrame": 64,    # 每帧包含的chirp数
    "isComplex": True,       # 是否为复数IQ数据（一般True）
    "sampleRate": 4e6,       # 采样率 (Hz)
    "chirpSlope": 29.982e12, # 调频斜率 (Hz/s)
    "adcSamplePeriod": 1/4e6,# 每个采样周期 (s)
    "framePeriod": 0.04,     # 每帧间隔 (s)，40ms
    "dataBytePerSample": 2,  # 每个采样占2字节
}

# ==========================
# 天线位置估计（仅示例）
# ==========================
def generate_antenna_positions(numRx, numTx):
    """
    生成虚拟天线位置表（单位：波长 λ）
    假设阵列为均匀线阵 (ULA)，间距为 λ/2
    """
    wavelength_spacing = 0.5  # 以 λ/2 为间距
    ant_positions = []
    for tx in range(numTx):
        for rx in range(numRx):
            ant_positions.append(tx * numRx * wavelength_spacing + rx * wavelength_spacing)
    return np.array(ant_positions)


# ==========================
# 读取单帧 bin 数据函数
# ==========================
def read_frame_bin(filename, cfg):
    """读取一个 bin 文件并解析为 [numSamples, numChirps, numAnt]"""
    numSamp = cfg["numSamplesPerChirp"]
    numChirp = cfg["numChirpsPerFrame"]
    numAnt = cfg["numRx"] * cfg["numTx"]
    numByte = cfg["dataBytePerSample"]

    len_total = numSamp * numChirp * numAnt * numByte
    with open(filename, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(raw_data) < len_total:
        raise ValueError(f"File {filename} too short. Expected {len_total}, got {len(raw_data)}")

    # 临时使用 int32 防止溢出
    FrameDataBuff = np.zeros(len_total // 2, dtype=np.int32)
    for i in range(len_total // 8):
        FrameDataBuff[i*4 + 0] = (raw_data[i*8 + 6] << 8) + raw_data[i*8 + 7]
        FrameDataBuff[i*4 + 1] = (raw_data[i*8 + 4] << 8) + raw_data[i*8 + 5]
        FrameDataBuff[i*4 + 2] = (raw_data[i*8 + 2] << 8) + raw_data[i*8 + 3]
        FrameDataBuff[i*4 + 3] = (raw_data[i*8 + 0] << 8) + raw_data[i*8 + 1]

    # 转换为有符号数
    FrameDataBuff[FrameDataBuff > 32767] -= 65536

    # 再转回 int16
    FrameDataBuff = FrameDataBuff.astype(np.int16)

    # # 每8字节解包为4个int16（高低字节反转）
    # FrameDataBuff = np.zeros(len_total // 2, dtype=np.int16)
    # for i in range(len_total // 8):
    #     FrameDataBuff[i*4 + 0] = (raw_data[i*8 + 6] << 8) + raw_data[i*8 + 7]
    #     FrameDataBuff[i*4 + 1] = (raw_data[i*8 + 4] << 8) + raw_data[i*8 + 5]
    #     FrameDataBuff[i*4 + 2] = (raw_data[i*8 + 2] << 8) + raw_data[i*8 + 3]
    #     FrameDataBuff[i*4 + 3] = (raw_data[i*8 + 0] << 8) + raw_data[i*8 + 1]

    # FrameDataBuff[FrameDataBuff > 32767] -= 65536

    # 重塑为 [numSamples, numChirps, numAnt]
    ADCBuf = np.zeros((numSamp, numChirp, numAnt), dtype=np.int16)
    for k in range(numChirp):
        for j in range(numAnt):
            start_idx = numSamp * (j + numAnt * k)
            ADCBuf[:, k, j] = FrameDataBuff[start_idx:start_idx + numSamp]

    return ADCBuf


# ==========================
# 主函数
# ==========================
def main():
    parser = argparse.ArgumentParser(description="Parse mmWave .bin radar data")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to .bin files directory")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to read")
    parser.add_argument("--save_mat", action="store_true", help="Save gBuf.mat output")
    parser.add_argument("--save_npz", action="store_true", help="Save gBuf.npz output")
    parser.add_argument("--plot", action="store_true", help="Show live waveform plot")
    args = parser.parse_args()

    cfg = CONFIG
    numAnt = cfg["numRx"] * cfg["numTx"]

    antenna_positions = generate_antenna_positions(cfg["numRx"], cfg["numTx"])
    print("Antenna virtual positions (λ units):", antenna_positions)

    gBuf = np.zeros(
        (cfg["numSamplesPerChirp"], cfg["numChirpsPerFrame"], numAnt, args.num_frames),
        dtype=np.int16,
    )

    timestamps = []

    for frame_idx in range(args.num_frames):
        filename = os.path.join(args.data_dir, f"flash_data_{frame_idx}.bin")
        if not os.path.exists(filename):
            print(f"[Warning] Missing file: {filename}")
            break

        ts = datetime.now().timestamp()
        timestamps.append(ts)

        ADCBuf = read_frame_bin(filename, cfg)
        gBuf[:, :, :, frame_idx] = ADCBuf

        if args.plot:
            plt.figure(1)
            plt.clf()
            offset = 3000
            for j in range(numAnt):
                # plt.plot(ADCBuf[:, 0, j] + offset * j)
                plt.plot(ADCBuf[:, 0, j].astype(np.float32) + offset * float(j))

            plt.title(f"Frame {frame_idx}")
            plt.ylim([-offset, offset * numAnt])
            plt.pause(0.05)

        print(f"[{frame_idx:03d}] Processed {filename}")

    print("All frames processed.")

    if args.save_mat:
        savemat("gBuf.mat", {"gBuf": gBuf, "timestamps": timestamps, "antenna_positions": antenna_positions})
        print("Saved gBuf.mat")
    if args.save_npz:
        np.savez(
            "gBuf.npz",
            gBuf=gBuf,
            timestamps=np.array(timestamps, dtype=np.float64),
            antenna_positions=antenna_positions,
        )
        print("Saved gBuf.npz")

    # 输出参数信息
    print("\nRadar Configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
