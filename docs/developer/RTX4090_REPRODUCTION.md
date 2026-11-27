# 双RTX4090服务器复现当前P1成果

更新：2026-11-27。功能基线为`f312f6a`所在的`feat/multimodal-p1`分支，包括交流进退、节目中接续、表情与话题、聊天记忆、图声身份、审核页、循声找人、侧方语义响应和熟人引介。声源方向按变化通知使用的现场验收见[第46项](ACCEPTANCE.md#46-方向保持循声找人及插话介绍2026-11-27)。

## 1. 本次服务器与复现范围

已通过原专用SSH通道只读核对：

| 项目 | 2026-11-27实际情况 |
| --- | --- |
| 登录与原工程 | `ubuntu@192.168.110.109:/home/ubuntu/project/`，保留原工程 |
| 系统／CPU架构 | Ubuntu22.04、x86_64 |
| 默认Python | 3.10.12；新工程使用独立Python3.12，不替换系统Python |
| GPU | 两张GeForce RTX4090，驱动报告595.80 |
| 旧Audio2Face SDK | `/home/ubuntu/project/third_party/Audio2Face-3D-SDK`，提交与本项目锁定的`1ca0f02535ed774f5dbcd724a31cd486368dc783`一致 |
| CUDA工具链 | 默认SSH环境找不到`nvcc`，需要检查旧SDK的依赖／隔离环境，不能据此说服务器不能运行CUDA |
| 建议新目录 | `/home/ubuntu/Vivis-delivery`，与原工程分开 |

先复现代码、模型及离线自检，再接外设验收。第一轮用一张卡运行A2F，另一张可留给其他任务；当前代码不是双卡并行推理程序。Qwen实时语音仍使用现有云端服务，增加GPU不会自动把它变成本地模型。

完整复现还需相机、阵列麦、音箱、声源串口和机器头。若这些仍接Orin，服务器只能先做软件／录音回放验证；当前成果没有“Orin采集控制、4090远程推理”的通用分布式部署入口，该方案需另做网络时钟、断线及控制权设计。

## 2. 取得正确分支

以下命令在4090服务器执行。目标目录已存在时先检查其内容，不要覆盖旧工程。

```bash
ssh ubuntu@192.168.110.109
git clone --branch feat/multimodal-p1 --single-branch \
  https://github.com/adniyanuo-ui/Vivis.git /home/ubuntu/Vivis-delivery
cd /home/ubuntu/Vivis-delivery
git status --short --branch
git log -1 --oneline
git merge-base --is-ancestor f312f6a HEAD
```

最后一条成功退出表示包含本次功能基线；不要只下载`main`或继续运行原工程的旧提交。若通过本地Git bundle先行导入，同样核对分支与基线，并把`origin`设回上述GitHub地址。

## 3. 建立独立Python环境，先做不接设备的检查

系统依赖由操作者安装；不要替换现有NVIDIA驱动，也不要直接把Orin的`.venv`复制过来。

```bash
sudo apt-get update
sudo apt-get install -y \
  git python3-venv python3-pip build-essential cmake ninja-build pkg-config \
  libportaudio2 alsa-utils libsndfile1 libspeexdsp1 \
  libeigen3-dev nlohmann-json3-dev zlib1g-dev sox ffmpeg

cd /home/ubuntu/Vivis-delivery
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv
.bootstrap/bin/uv python install 3.12
.bootstrap/bin/uv venv --python 3.12 --seed .venv
./tools/setup.sh
.venv/bin/python -m pip install 'cmake>=3.24,<4'
./tools/check_delivery.sh
```

`uv`仅用于取得独立Python，不改变系统默认解释器。基础环境与语音、视觉环境分别安装：不要把所有requirements合并到一个环境，当前基础环境和语音环境的SciPy版本不同。离线检查报可选模型／依赖缺项时，按下一节补齐后再运行；不能跳过失败并宣称完整复现。上述检查不打开真实设备。

## 4. 准备语音、视觉和声纹

仍在新目录执行，提前用Python3.12创建语音环境，避免实时语音安装器使用系统默认3.10：

```bash
.bootstrap/bin/uv venv --python 3.12 --seed outputs/speech/venv
./tools/setup_realtime_voice.sh --voice-preserve
.venv/bin/python tools/setup_p1.py
.venv/bin/python tools/setup_face_landmarker.py
.venv/bin/python tools/setup_speaker_observation.py
```

这些准备步骤安装依赖或下载清单中固定哈希的模型，不驱动机器头。提示音和固定短句缓存已在Git中的`assets/speech/`；保留现用音色与缓存，不重新随机生成一套。

人脸识别使用现有SeetaFace6；不能为了绕过编译改成SFace并沿用旧特征库。**当前`tools/setup_seetaface6.py`明确只允许aarch64，并写有ARM／NEON参数，不能在x86服务器直接照抄执行。**需要在新工程中适配该安装入口：保留锁定源码、两个补丁、模型哈希及原有算法；仅按x86构建对应CPU库，再编译`native/vision/seeta`探针。编译与质量探针通过后才进入熟人库验证。此项不是已经完成的4090现场验证。

## 5. A2F需要本机编译与本机引擎

| 可以复用 | 必须重新建立 |
| --- | --- |
| 锁定SDK源码、原始ONNX、模型JSON及哈希清单 | x86_64可执行程序、动态库、CMake构建目录 |
| 已有音频、口型缓存、节目JSON、机器人标定与限位配置 | 4090与本机TensorRT版本对应的`network.trt` |

RTX4090的CUDA计算能力为8.9，对应CMake架构参数89。当前`tools/setup_p0.py`写死87，属于Orin构建设置，需先给该入口增加明确的架构选择，再在服务器以89构建；保留Orin默认与原测试。仓库内的`native/a2f/CMakeLists.txt`和实时`a2f_chunk_probe`才是本次接入入口，不能仅运行旧服务器SDK的演示程序代替。

先检查旧SDK中实际安装的CUDA／TensorRT版本与路径，再选择匹配工具链。不要因为默认PATH没有`nvcc`就立即重装驱动，也不要直接安装“最新TensorRT”覆盖旧工程。SDK版本保持锁定；新编译产物应放在新项目的`outputs/a2f/`。TensorRT Python绑定与C++运行库要匹配，模型按`native/a2f/model_manifest.json`核验，重新生成引擎与`model.json`。

先指定一张空闲卡做模型构建和无设备音频回放，例如：

```bash
export CUDA_VISIBLE_DEVICES=0
```

当前C++程序使用可见设备0；若改为`CUDA_VISIBLE_DEVICES=1`，程序内的0对应物理第二张卡。选择后先验证真实A2F能输出52通道，再测试24k语音转16k、实际DAC进度同步和打断取消。固定音频缓存有口型不等于实时A2F已正常。

NVIDIA说明：RTX4090计算能力见[CUDA GPU列表](https://developer.nvidia.com/cuda-gpus)；引擎跨平台／硬件兼容限制见[TensorRT引擎兼容说明](https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/inference-library/engine-compatibility.html)。本项目没有启用可免除本次重建的跨平台引擎方案。

## 6. 单独迁移人物和记忆，密钥在服务器配置

| 内容 | 位置／处理 |
| --- | --- |
| 代码、配置、节目、固定短句与文档 | 随本分支克隆 |
| 熟人资料、原图、活动／候选照片和图像特征 | `private/people/`，经自有网络单独迁移 |
| 声纹活动／候选样本和特征 | `private/speakers/`，保留清单、锁定与来源，不只复制WAV |
| 近期完整音文和长期记忆 | `private/continuity/`及当前memory配置所指向的库；连同被引用的文件迁移 |
| 留言、业务账本和其他人物资料 | `private/`中相应库，保留人物ID及数据库关联 |
| 模型 | 可从原始模型来源按哈希下载；不要复制Orin二进制、引擎或整个outputs目录 |
| 百炼Key | 在新服务器本地重新配置，不进Git、不写入聊天或命令历史 |

停写后复制完整`private/`最容易保留关联。先由操作者正常停止Orin接待服务与审核服务，再做副本；运行中的SQLite不能只拷单个`.sqlite`文件而丢弃WAL。也可使用SQLite备份接口并核对关联音文文件。恢复后用审核页检查图片、录音、活动数量及文件路径；若清单含旧绝对路径，按具体字段做有验证的迁移，不全局盲目替换。

密钥配置入口见[开机服务说明](P1_AUTOSTART.md)；手动测试可用`--ask-key`安全输入。当前语音仍需百炼北京地域的相应模型权限。公开GitHub仓库不会保存原人物库、聊天录音或API Key。

## 7. 接外设后再运行完整交互

先逐一核对USB枚举、串口权限、相机by-id、输入／输出声卡名以及DOV回采。服务器设备名可能与Orin不同，使用原唯一配置源做本机适配；不改舵机PID、限位、扭矩和已接受的跟随方向来掩盖设备路径问题。

完整硬件具备、模型检查通过、操作者在场后，首次先前台手动运行：

```bash
cd /home/ubuntu/Vivis-delivery
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vivis.interaction.programs dialogue \
  --robot configs/robots/xiaoshuai/robot.json --ask-key
```

检查开机自检和回中通过，再依次测普通聊天与实时口型、表情、节目切换、身份和审核、无脸寻人、侧方问答、熟人引介。最后才安装服务器自己的systemd服务；不要把Orin的绝对路径单元直接复制过来。若通过systemd指定GPU，应在本机服务环境中单独设置，终端的`export`不会自动传给系统服务。

## 8. 可交给服务器Codex执行的迁移任务

在新工程目录打开Codex，交给它以下任务。它应先完成代码／模型适配和离线验证，真实设备部分留给操作者：

> 阅读AGENTS.md和本说明，复现feat/multimodal-p1且必须包含f312f6a。服务器是Ubuntu22.04双4090，原工程/home/ubuntu/project只读保留，当前新目录独立运行。先盘点已有CUDA/TensorRT和锁定SDK，使用独立Python3.12环境；适配A2F构建架构89和SeetaFace6的x86入口，保留Orin兼容、模型哈希、算法和运动保护。不要复制ARM虚拟环境、动态库或TensorRT引擎，不新增旁路LLM，不把Qwen擅自改为本地模型。完成基础检查、全量离线测试以及原始音频到真实4090 A2F的回放验证，清楚记录缺项与未验证内容。没有完整外设和人在现场，不启动硬件交互或开机服务。人物库经单独副本恢复后核对来源与路径，密钥只在本机安全配置。修改前保存本地基线，review后提交本地，不擅自推送。

本轮Orin固定版本的测试基线为1545项：1542通过、3项可选跳过；这不是已在4090服务器通过的结果。服务器需重新记录自己的环境版本、实际测试结果和GPU回放证据。
