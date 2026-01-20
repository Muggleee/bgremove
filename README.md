# Video Background Remover

使用 AI 自动去除视频背景，输出带透明通道的视频文件。

## 功能特点

- 基于 [rembg](https://github.com/danielgatis/rembg) 的 AI 背景去除
- 支持多种 AI 模型（通用、人像优化、动漫风格等）
- 输出 WebM (VP9) 或 MOV (ProRes 4444) 格式，支持透明通道
- 边缘清理：去除抠图后的边缘杂色
- 去绿溢色：处理绿幕拍摄的反光问题
- 分辨率调整：支持抠图前/后缩放
- 调试模式：快速预览第一帧效果
- 进度条显示处理进度

## 安装

```bash
# 克隆仓库
git clone https://github.com/Muggleee/bgremove.git
cd bgremove

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install onnxruntime
```

## 使用方法

### 基本用法

```bash
# 处理视频
python remove_bg.py input/video.mp4

# 指定输出文件
python remove_bg.py input/video.mp4 -o output/result.webm

# 使用人像优化模型
python remove_bg.py input/video.mp4 -m u2net_human_seg
```

### 调试模式

只处理第一帧，快速预览效果：

```bash
python remove_bg.py -d input/video.mp4 -o test.png
```

### 边缘清理

去除抠图后边缘残留的背景颜色：

```bash
# 启用边缘清理（默认 erode=1 + decontaminate）
python remove_bg.py --edge-clean input/video.mp4

# 自定义侵蚀像素
python remove_bg.py --erode 2 input/video.mp4

# 只用去污染不侵蚀
python remove_bg.py --erode 0 --decontaminate input/video.mp4
```

### 去绿溢色

处理绿幕拍摄时主体上的绿色反光：

```bash
# 启用去绿溢色
python remove_bg.py --despill input/video.mp4

# 调整强度 (0.0-2.0，默认 1.0)
python remove_bg.py --despill --spill-strength 0.8 input/video.mp4

# 使用 average 方法（更激进）
python remove_bg.py --despill --spill-method average input/video.mp4
```

去溢色方法说明：
- `max`（默认）：保守方式，`G = min(G, max(R, B))`
- `average`：激进方式，`G = min(G, (R+B)/2)`

### 分辨率调整

```bash
# 按比例缩放
python remove_bg.py --scale 0.5 input/video.mp4

# 指定高度（宽度按比例）
python remove_bg.py --height 720 input/video.mp4

# 指定宽度（高度按比例）
python remove_bg.py --width 1280 input/video.mp4

# 同时指定宽高（可能变形）
python remove_bg.py --width 1280 --height 720 input/video.mp4

# 在抠图后缩放（质量更好但更慢）
python remove_bg.py --height 720 --scale-after input/video.mp4

# 选择插值算法
python remove_bg.py --height 720 --interpolation lanczos input/video.mp4
```

插值算法：`nearest`、`linear`、`cubic`（默认）、`lanczos`

### 组合使用

```bash
# 完整处理流程：缩放 + 去溢色 + 边缘清理
python remove_bg.py --height 720 --despill --edge-clean input/video.mp4 -o output/result.webm

# 调试预览
python remove_bg.py -d --height 720 --despill --edge-clean input/video.mp4 -o preview.png
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

## 完整参数列表

```
python remove_bg.py [选项] 输入文件

位置参数:
  input                输入视频文件路径

选项:
  -o, --output         输出文件路径
  -m, --model          AI 模型 (默认: u2net)
  -d, --debug          调试模式，只处理第一帧

边缘处理:
  --edge-clean         启用边缘清理 (erode=1 + decontaminate)
  --erode N            边缘侵蚀像素数 (默认: 0)
  --decontaminate      启用边缘去污染

去绿溢色:
  --despill            启用去绿溢色
  --spill-strength F   去溢色强度 (默认: 1.0)
  --spill-method M     去溢色方法: max|average (默认: max)

分辨率调整:
  --scale F            缩放比例 (如 0.5)
  --width N            输出宽度
  --height N           输出高度
  --scale-after        在抠图后缩放 (默认抠图前)
  --interpolation M    插值算法: nearest|linear|cubic|lanczos (默认: cubic)
```

## 依赖

- Python 3.8+
- FFmpeg（需要安装在系统中）
- rembg
- opencv-python-headless
- numpy
- pillow
- tqdm
- scipy
- onnxruntime

## License

MIT
