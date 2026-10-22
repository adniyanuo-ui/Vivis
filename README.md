# EVA Audio2Face 与小白真机播放

语音 → NVIDIA Audio2Face → FaceMotion-52 → 61 通道 JSON → 飞特舵机。
当前已在 Ubuntu 22.04、RTX 4090 环境完成 Claire v3.0 视频生成和小白歌曲试播；2026-10-22 在 Jetson AGX Orin（Ubuntu 24.04 / ARM64）完成同一歌曲的完整真机播放，现场确认声音及面部动作正常。验证范围与通信记录见[审查记录](docs/CODE_REVIEW.md)。

## 目录与版本管理范围

- `tools/`：A2F 导出、表情与眼神编排、格式转换、预览和播放器；`xiaobai_runtime/` 为映射、串口与音频同步核心。
- `xiaobai/`：最终标定、61 通道顺序和三份必要配置。`playback_profile_0001_observation.json` 是一键播放配置；另两份供通用入口及启动/诊断工具使用。
- `tests/fixtures/`：合成帧和旧标定回归样例；旧标定只用于测试，真机使用 `xiaobai/xiaobai标定文件.json`。
- [设计文档](EVA_AUDIO2FACE_PIPELINE_V1.1.md)、[播放说明](PLAY_0001_XIAOBAI.md)、[审查记录](docs/CODE_REVIEW.md)。现有 `AGENTS.md` 保留初始版本，当前执行命令以本文为准。

`.gitignore` 采用白名单。模型、引擎、音视频、原始录制 CSV、Windows 程序、第三方源码、虚拟环境与 `outputs/`、`logs/` 均不提交；添加新源码需同步更新白名单。仓库不是完整媒体备份。

## 最小开发与验证环境

使用 Python 3.10、Git；系统需提供 PortAudio（Ubuntu 软件包 `libportaudio2`）。首次恢复：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements-xiaobai.txt
python3 tools/bootstrap_sources.py --component servo
.venv/bin/python -m unittest discover -s tools -p 'test_*.py'
git diff --check
```

依赖恢复工具核对 `tools/external-sources.json` 中的提交；已有目录版本不同或有已跟踪修改时退出，不覆盖。测试使用模拟设备与 PTY，不驱动机器人；未安装模型时仅跳过可选的网格左右绑定测试。

## 当前机器播放

另行恢复 `0001.wav` 和 `outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json`，再按[播放说明](PLAY_0001_XIAOBAI.md)准备串口与已归位颈部：

```bash
./play_0001_xiaobai.sh --check
./play_0001_xiaobai.sh                 # 默认 60fps
./play_0001_xiaobai.sh --fps 30
```

配置沿用现场确认的 2047 零权重基准、100% 幅度、位置限位、限速及反馈仅记录策略。`Ctrl+C` 停止发送；不会自动关闭颈部扭矩。设备串号、端口、音频设备与标定仅适用于这台小白，更换设备需重新核对。

## Jetson 上播放已有动作文件

ARM64 Ubuntu 24.04 / Python 3.12 可使用系统 NumPy 1.26.4、SciPy 1.11.4：

```bash
sudo apt-get install libportaudio2 python3-venv python3-numpy python3-scipy
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r tools/requirements-xiaobai.txt
python3 tools/bootstrap_sources.py --component servo
```

播放已有 JSON/WAV 不依赖 CUDA、TensorRT 或 Blender。仍需从原机器恢复上述两份歌曲文件，并核对是否为同一台小白及原标定。更换转接器后，先现场确认序列号，再显式设置 `XIAOBAI_USB_SERIAL`；默认仍匹配原转接器。驱动工具为调用 sudo 的用户授予临时串口权限。端口被其他程序占用时，需先正常停止对应程序，避免并发控制。

```bash
export XIAOBAI_USB_SERIAL=现场确认的序列号
python3 tools/xiaobai_switch_usb_driver.py status
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'
./play_0001_xiaobai.sh --check --audio-device 'USB Audio Device'
./play_0001_xiaobai.sh --max-seconds 15 --audio-device 'USB Audio Device' --audio-latency 0.2
```

音频设备名称需按本机枚举结果填写，并现场确认声音输出。USB 音频若出现 `output underflow`，可用 `--audio-latency 0.2` 请求 200 毫秒缓冲；动作继续跟随 DAC 输出时钟。端口通过已确认序列号自动定位，支持 USB 拔插后端口编号变化。`--check` 不播放音频，其通过不代表已确认扬声器可听，也不等于逐帧机械到位验证。Jetson 播放适配不代表已验证 Jetson 上的 A2F 推理；下面的 GPU 安装和构建辅助脚本仍含原 x86_64 环境路径及 GPU 配置，不能直接作为 Jetson 安装流程。

## 可选 A2F 与视频环境

`tools/requirements-env.txt` 保存完整 Python 环境版本；CUDA 12.9、TensorRT 10.13.3.9、Blender 4.5.3 和模型另行安装，二进制不入库。SDK/驱动分别通过 `bootstrap_sources.py --component a2f`、`--component driver` 恢复；驱动需针对当前内核编译，恢复命令不安装或切换驱动。

模型清单固定下载版本，例如：

```bash
.venv/bin/python tools/download_model_variant.py nvidia/Audio2Face-3D-v3.0 a2f-v3.0
source tools/env.sh
bash tools/build_a2f.sh
```

构建前需按 SDK 文档恢复其自身依赖并准备本仓库约定的本地 CUDA/TRT 目录；`install_cuda_local.py`、`install_trt_local.py` 和 `prepare_trt_layout.py` 保留本机安装辅助逻辑。歌曲编排脚本包含当前歌曲的时间点和素材路径，重跑需恢复中间产物或重新生成。此次审查未验证从空机器一键重建完整 GPU/渲染环境。
