# Vivis 墨甲机电仿生头开发平台

版本：开发者整理版0.1，2026-10-26。运动基线：`994a13b`；本分支三个新入口待现场验收。

面向具备AI工程或二次开发能力的团队，提供本体控制代码、最终标定、完整示例及教程。客户负责音频、文本、情绪与业务逻辑；开发包负责表情到本体的执行链路及视觉跟踪示例。Jetson Orin为可选主控，不是固定依赖。

阅读顺序：[快速上手](docs/developer/QUICKSTART.md) → [整体架构](docs/developer/ARCHITECTURE.md) → [本体与标定](docs/developer/HARDWARE_AND_CALIBRATION.md) → [数据与接口](docs/developer/DATA_AND_CONTROL_API.md)。

## 1. 三个示例

| 示例 | 教程 | 播放阶段 |
|---|---|---|
| 1 | [面部控制](examples/01_face_control/README.md) | 音频＋表情，不启动相机，眼球/颈部保持 |
| 2 | [面部＋眼球](examples/02_face_eye_tracking/README.md) | 增加横向双眼ID8/23跟踪，颈部保持 |
| 3 | [面部＋眼球＋颈部](examples/03_face_eye_neck_tracking/README.md) | 增加ID32协同，ID7/22/33/34保持 |

```bash
./examples/01_face_control/run.sh --check
./examples/02_face_eye_tracking/run.sh --check
./examples/03_face_eye_neck_tracking/run.sh --check

./examples/01_face_control/run.sh
./examples/02_face_eye_tracking/run.sh
./examples/03_face_eye_neck_tracking/run.sh
```

后3条请逐个运行。默认wz2、1遍；`--work 0001`切换完整歌曲。每遍独立准备、35轴速度60缓慢回中后，从音频0开始；“保持”指回中后的播放阶段。示例2/3运行期间监控为`http://127.0.0.1:8765/`。自行启动的视觉服务在结束或异常后关闭；外部服务只在配置哈希匹配时复用，不由示例关闭。

## 2. 目录与配置来源

| 路径 | 职责 |
|---|---|
| `src/vivis/motion/` | 通道适配、表情融合、标定映射及控制权 |
| `src/vivis/runtime/` | 配置解析、会话、回中、音频同步与调度 |
| `src/vivis/hardware/feetech/` | 协议、SDK适配及独立调试工具 |
| `src/vivis/vision/` | 相机、人脸检测、眼颈协调与预览 |
| `configs/robots/mojia/` | 当前标定、35轴参数、跟踪、回中和反馈策略 |
| `configs/schemas/` | 61通道名称及顺序，前52维为基础表情 |
| `assets/works.json` | 作品素材位置和SHA256，不保存电机参数 |
| `examples/` | 三份可独立运行的示例代码和教程 |
| `tools/` | 上游A2F实验、素材制作、诊断及旧工具兼容入口 |
| `outputs/` | 本地运行日志、有效配置及生成素材，不入Git |
| `mojia/`、`xiaobai/` | 历史资源及兼容路径，新入口不读取旧作品profile的电机参数 |

电机参数以`hardware.json`为准，运行profile、启动设置和跟踪器由同一表生成；软件告警以`telemetry.json`为准。最终标定为`calibration.json`，旧路径是兼容符号链接。[配置关系](docs/developer/DATA_AND_CONTROL_API.md#4-唯一配置来源)。

## 3. 当前边界

- 已验证环境：Jetson AGX Orin、ARM64、Ubuntu24.04、Python3.12；其他主机和操作系统待真机验证。核心播放不需要CUDA，上游A2F单独部署。
- 实际默认60Hz/1Mbps；500kbps仅为候选，尚未启用。`--baud`只改主机速率，不能代替35轴切换和断电重启核验。
- 保留14个口型相关轴速度1200、其余映射轴1000、口型增益1.0；软件加减速仅32/33/34为64，默认关闭纵向眼球跟踪。
- 默认右眼相机随ID23转动；左眼节点需明确配置，支持选择一只相机，不包含双目融合。
- 示例播放已有音频和表情序列，不代表已验证Jetson实时A2F推理；当前直接飞特协议，ROS2、自模型与学习逆解为扩展方向。

## 4. 安装与检查

完整步骤见[快速上手](docs/developer/QUICKSTART.md)。Git不包含媒体、模型、第三方源码和虚拟环境，按[素材说明](assets/README.md)恢复。当前使用保留源码和配置目录的editable安装，不承诺独立wheel分发。

```bash
.venv/bin/python -m pip install -e '.[vision]'
PYTHONPATH=src .venv/bin/python -m vivis.runtime.assets
.venv/bin/python -m unittest discover -s tools -p 'test_*.py'
git diff --check
```

原入口兼容同一核心：`./play_mojia_full_show.sh --repeat 1`、`./play_mojia_wz2.sh --repeat 1`。前者默认0001三遍，后者默认wz2一遍。旧高级`--profile/--targets`入口仍为`tools/mojia_performance.py`。

[EVA上游设计](EVA_AUDIO2FACE_PIPELINE_V1.1.md)｜[旧README及试验入口](docs/history/LEGACY_README_20261025.md)｜[历史审查](docs/CODE_REVIEW.md)。历史文档记录当时状态，当前用法以上述开发者教程为准。

本轮变更与离线验证：[整理审查记录](docs/developer/REVIEW_20261026.md)；现场结果填写[验收表](docs/developer/ACCEPTANCE.md)。
