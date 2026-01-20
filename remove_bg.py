#!/usr/bin/env python3
"""
视频背景去除工具
使用 rembg 去除视频中的背景，输出带透明通道的视频
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from tqdm import tqdm


def process_video(input_path: str, output_path: str, model: str = "u2net"):
    """
    处理视频，去除背景
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        model: rembg 模型名称
    """
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        sys.exit(1)
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps}")
    print(f"  总帧数: {total_frames}")
    print(f"  使用模型: {model}")
    
    # 创建临时目录存放帧
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir)
    
    try:
        # 处理每一帧
        print("\n正在处理视频帧...")
        frame_idx = 0
        
        with tqdm(total=total_frames, desc="去除背景") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # BGR 转 RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # 去除背景
                result = remove(pil_image, session_name=model)
                
                # 保存为 PNG（保留透明通道）
                output_frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                result.save(output_frame_path)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # 使用 ffmpeg 合成视频（支持透明通道）
        print("\n正在合成视频...")
        
        # 输出为 WebM (VP9) 或 MOV (ProRes) 格式以支持透明
        output_ext = Path(output_path).suffix.lower()
        
        if output_ext == ".webm":
            # WebM with VP9 (支持透明)
            ffmpeg_cmd = (
                f'ffmpeg -y -framerate {fps} -i "{frames_dir}/frame_%06d.png" '
                f'-c:v libvpx-vp9 -pix_fmt yuva420p -b:v 2M '
                f'"{output_path}"'
            )
        elif output_ext == ".mov":
            # MOV with ProRes 4444 (支持透明)
            ffmpeg_cmd = (
                f'ffmpeg -y -framerate {fps} -i "{frames_dir}/frame_%06d.png" '
                f'-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le '
                f'"{output_path}"'
            )
        else:
            # 默认输出 WebM
            if not output_path.endswith(".webm"):
                output_path = str(Path(output_path).with_suffix(".webm"))
            ffmpeg_cmd = (
                f'ffmpeg -y -framerate {fps} -i "{frames_dir}/frame_%06d.png" '
                f'-c:v libvpx-vp9 -pix_fmt yuva420p -b:v 2M '
                f'"{output_path}"'
            )
        
        os.system(ffmpeg_cmd)
        
        print(f"\n完成! 输出文件: {output_path}")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="视频背景去除工具 - 去除视频背景并输出透明视频"
    )
    parser.add_argument(
        "input",
        help="输入视频文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出视频文件路径 (默认: 输入文件名_nobg.webm)",
        default=None
    )
    parser.add_argument(
        "-m", "--model",
        help="rembg 模型 (默认: u2net)",
        default="u2net",
        choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime"]
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 {args.input}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_nobg.webm")
    
    process_video(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
