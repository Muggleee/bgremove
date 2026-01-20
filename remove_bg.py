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
from scipy import ndimage


def clean_edge(image: Image.Image, erode_size: int = 1, decontaminate: bool = False) -> Image.Image:
    """
    清理抠图边缘的颜色溢出
    
    Args:
        image: 带透明通道的 PIL Image (RGBA)
        erode_size: 边缘收缩像素数
        decontaminate: 是否进行颜色净化
    
    Returns:
        处理后的图像
    """
    # 转换为 numpy 数组
    img_array = np.array(image)
    
    if img_array.shape[2] != 4:
        return image  # 没有 alpha 通道，直接返回
    
    rgb = img_array[:, :, :3].astype(np.float32)
    alpha = img_array[:, :, 3].astype(np.float32)
    
    # 阶段1：边缘收缩（腐蚀 alpha 通道）
    if erode_size > 0:
        # 创建腐蚀核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size * 2 + 1, erode_size * 2 + 1))
        # 对 alpha 通道进行腐蚀
        alpha_eroded = cv2.erode(alpha, kernel, iterations=1)
        alpha = alpha_eroded
    
    # 阶段2：颜色净化
    if decontaminate:
        # 找到边缘区域（半透明像素）
        edge_mask = (alpha > 0) & (alpha < 255)
        
        # 找到完全不透明的区域
        opaque_mask = alpha >= 250
        
        if np.any(edge_mask) and np.any(opaque_mask):
            # 对每个颜色通道进行处理
            for c in range(3):
                channel = rgb[:, :, c]
                
                # 用最近的不透明像素颜色填充边缘
                # 创建一个只包含不透明区域颜色的图像
                opaque_colors = np.where(opaque_mask, channel, 0)
                
                # 使用距离变换找到最近的不透明像素
                dist, indices = ndimage.distance_transform_edt(~opaque_mask, return_indices=True)
                
                # 用最近不透明像素的颜色替换边缘颜色
                nearest_colors = channel[indices[0], indices[1]]
                
                # 只在边缘区域替换颜色
                rgb[:, :, c] = np.where(edge_mask, nearest_colors, channel)
    
    # 合并结果
    result = np.zeros_like(img_array)
    result[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result)


def process_video(input_path: str, output_path: str, model: str = "u2net", debug: bool = False,
                  edge_clean: bool = False, erode_size: int = 1, decontaminate: bool = False):
    """
    处理视频，去除背景
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        model: rembg 模型名称
        debug: 调试模式，只处理第一帧并输出 PNG
        edge_clean: 是否清理边缘
        erode_size: 边缘收缩像素数
        decontaminate: 是否进行颜色净化
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
    if edge_clean or erode_size > 0 or decontaminate:
        print(f"  边缘处理: erode={erode_size}px, decontaminate={decontaminate}")
    if debug:
        print(f"  调试模式: 只处理第一帧")
    
    # 调试模式：只处理第一帧
    if debug:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取视频帧")
            cap.release()
            sys.exit(1)
        
        # BGR 转 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        print("\n正在处理第一帧...")
        # 去除背景
        result = remove(pil_image, session_name=model)
        
        # 边缘清理
        if edge_clean or erode_size > 0 or decontaminate:
            result = clean_edge(result, erode_size=erode_size, decontaminate=decontaminate)
        
        # 保存为 PNG
        debug_output = output_path if output_path.endswith(".png") else str(Path(output_path).with_suffix(".png"))
        result.save(debug_output)
        
        cap.release()
        print(f"\n调试完成! 输出文件: {debug_output}")
        return
    
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
                
                # 边缘清理
                if edge_clean or erode_size > 0 or decontaminate:
                    result = clean_edge(result, erode_size=erode_size, decontaminate=decontaminate)
                
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
    parser.add_argument(
        "-d", "--debug",
        help="调试模式：只处理第一帧并输出 PNG",
        action="store_true"
    )
    parser.add_argument(
        "--edge-clean",
        help="启用边缘清理（默认: erode=1 + decontaminate）",
        action="store_true"
    )
    parser.add_argument(
        "--erode",
        help="边缘收缩像素数 (默认: 1)",
        type=int,
        default=0
    )
    parser.add_argument(
        "--decontaminate",
        help="启用颜色净化，去除边缘颜色溢出",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 {args.input}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output is None:
        input_path = Path(args.input)
        if args.debug:
            args.output = str(input_path.parent / f"{input_path.stem}_debug.png")
        else:
            args.output = str(input_path.parent / f"{input_path.stem}_nobg.webm")
    
    # 处理 edge_clean 默认参数：--edge-clean 默认启用 erode=1 + decontaminate
    erode_size = args.erode if args.erode > 0 else (1 if args.edge_clean else 0)
    decontaminate = args.decontaminate or args.edge_clean
    
    process_video(
        args.input, 
        args.output, 
        args.model, 
        args.debug,
        edge_clean=args.edge_clean,
        erode_size=erode_size,
        decontaminate=decontaminate
    )


if __name__ == "__main__":
    main()
