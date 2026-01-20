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
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove
from tqdm import tqdm
from scipy import ndimage


# 插值算法映射
INTERPOLATION_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "linear": Image.Resampling.BILINEAR,
    "cubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

# 默认 PAGConvertor 路径
DEFAULT_PAG_TOOL = "/Users/liziyang/Downloads/PAGConvertor/PAGConvertor"


def resize_image(image: Image.Image, scale: float = None, width: int = None, height: int = None,
                 interpolation: str = "cubic") -> Image.Image:
    """
    调整图像分辨率
    
    Args:
        image: PIL Image
        scale: 缩放比例 (如 0.5 = 缩小一半, 2.0 = 放大两倍)
        width: 目标宽度 (如果只指定宽度，高度按比例计算)
        height: 目标高度 (如果只指定高度，宽度按比例计算)
        interpolation: 插值算法 (nearest/linear/cubic/lanczos)
    
    Returns:
        缩放后的图像
    """
    orig_width, orig_height = image.size
    
    # 获取插值方法
    resample = INTERPOLATION_METHODS.get(interpolation, Image.Resampling.BICUBIC)
    
    # 计算目标尺寸
    if scale is not None:
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    elif width is not None and height is not None:
        # 同时指定宽高，可能变形
        new_width = width
        new_height = height
    elif width is not None:
        # 只指定宽度，保持宽高比
        new_width = width
        new_height = int(orig_height * (width / orig_width))
    elif height is not None:
        # 只指定高度，保持宽高比
        new_height = height
        new_width = int(orig_width * (height / orig_height))
    else:
        # 没有指定任何缩放参数，返回原图
        return image
    
    # 确保尺寸至少为1
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    return image.resize((new_width, new_height), resample=resample)


def despill_green(image: Image.Image, strength: float = 1.0, method: str = "average") -> Image.Image:
    """
    去除绿幕溢色（Green Spill Removal）
    
    Args:
        image: 带透明通道的 PIL Image (RGBA)
        strength: 去溢色强度 (0.0-1.0)
        method: 去溢色方法
            - "average": 用红蓝通道平均值替换绿色溢出
            - "max": 用红蓝通道最大值限制绿色
    
    Returns:
        处理后的图像
    """
    img_array = np.array(image)
    
    if len(img_array.shape) < 3:
        return image
    
    # 分离通道
    if img_array.shape[2] == 4:
        r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]
        has_alpha = True
    else:
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        a = None
        has_alpha = False
    
    # 转换为 float 进行计算
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    
    if method == "average":
        # 方法1：用红蓝平均值替换绿色溢出部分
        # 计算绿色溢出量：绿色超过红蓝平均值的部分
        rb_avg = (r + b) / 2
        spill = np.maximum(g - rb_avg, 0)
        
        # 按强度去除溢色
        g_new = g - spill * strength
        
    elif method == "max":
        # 方法2：限制绿色不超过红蓝最大值
        rb_max = np.maximum(r, b)
        spill = np.maximum(g - rb_max, 0)
        
        # 按强度去除溢色
        g_new = g - spill * strength
    
    else:
        g_new = g
    
    # 确保值在有效范围内
    g_new = np.clip(g_new, 0, 255).astype(np.uint8)
    r = np.clip(r, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    # 重新组合图像
    if has_alpha:
        result = np.stack([r, g_new, b, a], axis=2)
    else:
        result = np.stack([r, g_new, b], axis=2)
    
    return Image.fromarray(result)


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
                  edge_clean: bool = False, erode_size: int = 1, decontaminate: bool = False,
                  despill: bool = False, spill_strength: float = 1.0, spill_method: str = "average",
                  scale: float = None, width: int = None, height: int = None,
                  scale_before: bool = False, interpolation: str = "cubic",
                  output_pag: bool = False, pag_tool: str = None, keep_frames: bool = False):
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
        despill: 是否去除绿溢色
        spill_strength: 去绿溢色强度
        spill_method: 去绿溢色方法
        scale: 缩放比例
        width: 目标宽度
        height: 目标高度
        scale_before: 是否在抠图前缩放
        interpolation: 插值算法
        output_pag: 是否输出 PAG 文件
        pag_tool: PAGConvertor 工具路径
        keep_frames: 是否保留 PNG 序列帧
    """
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        sys.exit(1)
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 判断是否需要缩放
    need_resize = scale is not None or width is not None or height is not None
    
    print(f"视频信息:")
    print(f"  分辨率: {width_orig}x{height_orig}")
    print(f"  帧率: {fps}")
    print(f"  总帧数: {total_frames}")
    print(f"  使用模型: {model}")
    if need_resize:
        # 计算目标尺寸用于显示
        if scale is not None:
            target_w, target_h = int(width_orig * scale), int(height_orig * scale)
        elif width is not None and height is not None:
            target_w, target_h = width, height
        elif width is not None:
            target_w = width
            target_h = int(height_orig * (width / width_orig))
        else:
            target_h = height
            target_w = int(width_orig * (height / height_orig))
        print(f"  缩放: {width_orig}x{height_orig} → {target_w}x{target_h} ({'抠图前' if scale_before else '抠图后'}, {interpolation})")
    if despill:
        print(f"  去绿溢色: strength={spill_strength}, method={spill_method}")
    if edge_clean or erode_size > 0 or decontaminate:
        print(f"  边缘处理: erode={erode_size}px, decontaminate={decontaminate}")
    if debug:
        print(f"  调试模式: 只处理第一帧")
    if output_pag:
        print(f"  输出 PAG: {pag_tool or DEFAULT_PAG_TOOL}")
    if keep_frames:
        print(f"  保留序列帧: 是")
    
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
        
        # 抠图前缩放
        if need_resize and scale_before:
            pil_image = resize_image(pil_image, scale=scale, width=width, height=height, interpolation=interpolation)
        
        # 去除背景
        result = remove(pil_image, session_name=model)
        
        # 去绿溢色（在边缘清理之前）
        if despill:
            result = despill_green(result, strength=spill_strength, method=spill_method)
        
        # 边缘清理
        if edge_clean or erode_size > 0 or decontaminate:
            result = clean_edge(result, erode_size=erode_size, decontaminate=decontaminate)
        
        # 抠图后缩放
        if need_resize and not scale_before:
            result = resize_image(result, scale=scale, width=width, height=height, interpolation=interpolation)
        
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
                
                # 抠图前缩放
                if need_resize and scale_before:
                    pil_image = resize_image(pil_image, scale=scale, width=width, height=height, interpolation=interpolation)
                
                # 去除背景
                result = remove(pil_image, session_name=model)
                
                # 去绿溢色（在边缘清理之前）
                if despill:
                    result = despill_green(result, strength=spill_strength, method=spill_method)
                
                # 边缘清理
                if edge_clean or erode_size > 0 or decontaminate:
                    result = clean_edge(result, erode_size=erode_size, decontaminate=decontaminate)
                
                # 抠图后缩放
                if need_resize and not scale_before:
                    result = resize_image(result, scale=scale, width=width, height=height, interpolation=interpolation)
                
                # 保存为 PNG（保留透明通道）
                output_frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
                result.save(output_frame_path)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # 判断输出格式
        output_ext = Path(output_path).suffix.lower()
        output_video = output_ext not in [".pag"]  # 是否需要输出视频
        
        # 使用 ffmpeg 合成视频（支持透明通道）
        if output_video:
            print("\n正在合成视频...")
            
            # 输出为 WebM (VP9) 或 MOV (ProRes) 格式以支持透明
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
            print(f"视频输出: {output_path}")
        
        # PAG 转换
        if output_pag or output_ext == ".pag":
            print("\n正在转换 PAG...")
            pag_converter = pag_tool or DEFAULT_PAG_TOOL
            
            # 检查 PAGConvertor 是否存在
            if not os.path.exists(pag_converter):
                print(f"错误: PAGConvertor 不存在: {pag_converter}")
                print("请下载 PAGConvertor 并指定路径: --pag-tool <path>")
            else:
                # PAGConvertor 用法: PAGConvertor <序列帧文件夹> <帧率>
                pag_output = str(Path(output_path).with_suffix(".pag"))
                
                try:
                    result = subprocess.run(
                        [pag_converter, frames_dir, str(int(fps))],
                        capture_output=True,
                        text=True
                    )
                    
                    # PAGConvertor 会在 frames_dir 同级目录生成 frames.pag
                    generated_pag = os.path.join(temp_dir, "frames.pag")
                    
                    if os.path.exists(generated_pag):
                        # 移动到目标位置
                        shutil.move(generated_pag, pag_output)
                        print(f"PAG 输出: {pag_output}")
                    else:
                        print(f"警告: PAG 文件未生成")
                        if result.stderr:
                            print(f"错误信息: {result.stderr}")
                except Exception as e:
                    print(f"PAG 转换失败: {e}")
        
        # 保留序列帧
        if keep_frames:
            frames_output = str(Path(output_path).with_suffix("")) + "_frames"
            if os.path.exists(frames_output):
                shutil.rmtree(frames_output)
            shutil.copytree(frames_dir, frames_output)
            print(f"序列帧输出: {frames_output}/")
        
        print(f"\n完成!")
        
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
    parser.add_argument(
        "--despill",
        help="启用去绿溢色，去除主体上的绿幕反光",
        action="store_true"
    )
    parser.add_argument(
        "--spill-strength",
        help="去绿溢色强度 (0.0-1.0, 默认: 1.0)",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--spill-method",
        help="去绿溢色方法 (默认: max)",
        default="max",
        choices=["average", "max"]
    )
    parser.add_argument(
        "--scale",
        help="缩放比例 (如 0.5=缩小一半, 2.0=放大两倍)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--width",
        help="输出宽度 (高度按比例计算，除非同时指定 --height)",
        type=int,
        default=None
    )
    parser.add_argument(
        "--height",
        help="输出高度 (宽度按比例计算，除非同时指定 --width)",
        type=int,
        default=None
    )
    parser.add_argument(
        "--scale-after",
        help="在抠图后缩放 (默认在抠图前缩放以加快速度)",
        action="store_true"
    )
    parser.add_argument(
        "--interpolation",
        help="缩放插值算法 (默认: cubic)",
        default="cubic",
        choices=["nearest", "linear", "cubic", "lanczos"]
    )
    parser.add_argument(
        "--pag",
        help="输出 PAG 格式",
        action="store_true"
    )
    parser.add_argument(
        "--pag-tool",
        help=f"PAGConvertor 路径 (默认: {DEFAULT_PAG_TOOL})",
        default=None
    )
    parser.add_argument(
        "--keep-frames",
        help="保留 PNG 序列帧",
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
        elif args.pag and not args.keep_frames:
            # 只输出 PAG 时，默认扩展名为 .pag
            args.output = str(input_path.parent / f"{input_path.stem}_nobg.pag")
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
        decontaminate=decontaminate,
        despill=args.despill,
        spill_strength=args.spill_strength,
        spill_method=args.spill_method,
        scale=args.scale,
        width=args.width,
        height=args.height,
        scale_before=not args.scale_after,
        interpolation=args.interpolation,
        output_pag=args.pag,
        pag_tool=args.pag_tool,
        keep_frames=args.keep_frames
    )


if __name__ == "__main__":
    main()
