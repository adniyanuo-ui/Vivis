# Vivis 墨甲机电仿生头开发平台

版本0.2｜面向开发者的独立源码包。客户负责上游音频、文本、AI和业务逻辑，本包提供最终标定、表情到本体的控制链路及三种完整示例。Jetson Orin为可选主控。

## 1. 安装后直接运行

源码、两份作品的音频/目标轨迹、YuNet模型和必要的飞特SDK源码均随Git提供。无需从旧工程复制文件，无需另行clone SDK或下载模型。Python依赖通过pip安装，系统需Python3.10–3.12和PortAudio；已验证主机为Jetson AGX Orin、Ubuntu24.04、Python3.12。

```bash
# 新Ubuntu主机缺少系统依赖时先安装；已有环境不必重复执行。
sudo apt-get install python3-venv libportaudio2 alsa-utils
./tools/setup.sh
```

安装脚本建立独立`.venv`、安装`.[vision]`、核验素材和模型，并检查三个入口。安装/离线检查不打开串口、音频或相机。只跑面部示例也可手动执行`python3 -m venv .venv`和`.venv/bin/python -m pip install -e .`。

核对设备配置后，逐个运行：

```bash
./examples/01_face_control/run.sh
./examples/02_face_eye_tracking/run.sh
./examples/03_face_eye_neck_tracking/run.sh
```

默认wz2完整一遍，切换完整歌曲加`--work 0001`。每遍独立准备和35轴缓慢回中，然后从音频0开始。示例2/3运行期间监控为`http://127.0.0.1:8765/`。

## 2. 三个示例

| 示例 | 教程 | 播放阶段 |
|---|---|---|
| 1 | [面部控制](examples/01_face_control/README.md) | 音频＋表情，不启动相机；眼球/颈部保持 |
| 2 | [面部＋眼球](examples/02_face_eye_tracking/README.md) | 增加横向双眼ID8/23跟踪，颈部保持 |
| 3 | [面部＋眼球＋颈部](examples/03_face_eye_neck_tracking/README.md) | 增加ID32协同；7/22/33/34保持 |

准备/回中会涉及35轴，“保持”指正式播放阶段。一次只运行一个示例。示例自行启动的视觉服务在结束/异常后关闭；复用外部视觉服务需配置、目标协议、相机方向一致，且图像仍新鲜。

## 3. 干净的目录

```text
Vivis/
├── README.md
├── pyproject.toml
├── src/vivis/
│   ├── motion/                 # 适配、融合、标定映射、控制权
│   ├── runtime/                # 音频同步、调度、会话、日志
│   ├── vision/                 # 相机、人脸检测、眼颈协调
│   └── hardware/feetech/        # 协议与随包SDK子集
├── configs/robots/mojia/        # 唯一设备参数与最终标定
├── configs/schemas/             # 通道名称及顺序
├── examples/                   # 三个示例及教程
├── assets/                     # 两份作品、模型、来源清单
├── docs/developer/              # 架构、本体、接口、排障
├── tools/                      # 安装、素材/标定检查、作品导入
└── tests/                      # 控制核心与三模式回归
```

本分支不包含小白目录、历史播放入口、旧配置副本、A2F训练/渲染实验或历史文档归档。历史版本保留在Git原有提交及备份分支；运行不依赖它们。

## 4. 阅读与修改

[快速上手](docs/developer/QUICKSTART.md) → [整体架构](docs/developer/ARCHITECTURE.md)／[离线架构图](docs/developer/ARCHITECTURE.html) → [本体与标定](docs/developer/HARDWARE_AND_CALIBRATION.md) → [数据与接口](docs/developer/DATA_AND_CONTROL_API.md)。

- 改算法：`src/vivis/`。editable安装后改源码即生效，示例不复制控制算法。
- 改设备参数：`configs/robots/mojia/`。运行profile、启动设置及跟踪器使用同一硬件表。
- 可选颈部平滑：[上游轨迹规划](docs/developer/UPSTREAM_TRAJECTORY_PLANNING.md)，默认关闭；示例3用`--upstream-trajectory on/off`选择，不影响其他轴。
- 改输入：用自己的音频与61通道文件[导入新作品](docs/developer/DATA_AND_CONTROL_API.md#3-接入自己的音频与表情序列)。
- 核验素材：`.venv/bin/python tools/check_assets.py`；核验标定覆盖：`.venv/bin/python tools/check_calibration.py`。
- 扩展上游：客户可以接入LLM、TTS、A2F等；已有示例不依赖CUDA，也不等于实时A2F推理已验收。

默认仍60Hz/1Mbps；500kbps候选尚未启用。设备节点、相机侧和音箱按实际本体核对。示例3已完成wz2一遍真机运行，用户反馈成功；随后用户更换ID32并恢复同款PID，颈部抖动尚未解决。示例1/2及新参数的表现仍需[现场验收](docs/developer/ACCEPTANCE.md)。

10-27移除软件固定加减速后，用户完整播放0001三遍：面部正常，32号启动抖动、回中过快。随后按要求将ID32的SRAM41设为128，并加入150ms颈部目标低通和至少3秒平滑回中，其他参数及素材不变。该最新组合尚未真机复测。

## 5. 回归与交接

```bash
./tools/check_delivery.sh
```

测试使用模拟设备、PTY和合成图像，不驱动真机。运行日志写入被忽略的`outputs/`。不得将控制频率或离线通过描述为机械每帧到位。第三方源码、模型和许可证位置见[素材与依赖](assets/README.md)及`assets/third_party.json`。

离线自检覆盖依赖、资源哈希、标定、六组“示例/作品”入口及模拟设备回归，不打开机器头。已完成从本地Git重新clone、新建`.venv`安装和独立运行核验，源码、SDK、模型和素材均位于新克隆内。最新审查结论及待验收项见[交付自检](docs/developer/DELIVERY_REVIEW.md)；对外交付前须推送相应提交并从GitHub重新clone核对，原Git历史仍保留。
