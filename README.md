# Video Background Remover

使用 AI 自动去除视频背景，输出带透明通道的视频文件。

## 功能特点

- 基于 [rembg](https://github.com/danielgatis/rembg) 的 AI 背景去除
- 支持多种 AI 模型（通用、人像优化、动漫风格等）
- 输出 WebM (VP9) 或 MOV (ProRes 4444) 格式，支持透明通道
- 进度条显示处理进度

## 安装

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/bgremove.git
cd bgremove

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install onnxruntime
```

## 使用方法

```bash
# 基本用法
python remove_bg.py input_video.mp4

# 指定输出文件
python remove_bg.py input_video.mp4 -o output.webm

# 使用人像优化模型
python remove_bg.py input_video.mp4 -m u2net_human_seg

# 使用动漫风格模型
python remove_bg.py input_video.mp4 -m isnet-anime
```

## 支持的模型

| 模型名称 | 说明 |
|---------|------|
| `u2net` | 通用模型（默认） |
| `u2netp` | 轻量级通用模型 |
| `u2net_human_seg` | 人像优化 |
| `u2net_cloth_seg` | 服装分割 |
| `silueta` | 轮廓提取 |
| `isnet-general-use` | 高质量通用 |
| `isnet-anime` | 动漫风格 |

## 输出格式

- `.webm` - VP9 编码，支持透明（推荐，文件较小）
- `.mov` - ProRes 4444 编码，支持透明（高质量，文件较大）

## 依赖

- Python 3.8+
- FFmpeg（需要安装在系统中）
- rembg
- opencv-python-headless
- numpy
- pillow
- tqdm
- onnxruntime

## License

MIT
