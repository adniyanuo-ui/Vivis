# EVA Audio2Face 与小白真机播放

语音 → NVIDIA Audio2Face → FaceMotion-52 → 61 通道 JSON → 飞特舵机。
当前已在 Ubuntu 22.04、RTX 4090 环境完成 Claire v3.0 视频生成和小白歌曲试播；2026-10-22 在 Jetson AGX Orin（Ubuntu 24.04 / ARM64）完成同一歌曲的完整真机播放，现场确认声音及面部动作正常。验证范围与通信记录见[审查记录](docs/CODE_REVIEW.md)。

## 目录与版本管理范围

- `tools/`：A2F 导出、表情与眼神编排、格式转换、预览和播放器；`xiaobai_runtime/` 为映射、串口与音频同步核心。
- `xiaobai/`：最终标定、61 通道顺序、三份播放配置和人脸跟踪配置。`playback_profile_0001_observation.json` 是一键播放配置；另两份供通用入口及启动/诊断工具使用。
- `tests/fixtures/`：合成帧和旧标定回归样例；旧标定只用于测试，真机使用 `xiaobai/xiaobai标定文件.json`。
- [设计文档](EVA_AUDIO2FACE_PIPELINE_V1.1.md)、[播放说明](PLAY_0001_XIAOBAI.md)、[审查记录](docs/CODE_REVIEW.md)。现有 `AGENTS.md` 保留初始版本，当前执行命令以本文为准。
- [人脸水平跟随方案与验收](docs/PERSON_GAZE_V1.0.md)：USB 摄像头选择最大人脸，眼球 ID 8、23 先响应，颈部 ID 32 随后分担偏转；已保存现场认可的三倍转头速度配置。

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

## 小白人物跟踪（小白专用）

使用已认可的眼头跟随效果，先安装视觉依赖并恢复固定版本模型，再分别在两个终端启动：

```bash
.venv/bin/python -m pip install -r tools/requirements-vision.txt
.venv/bin/python tools/download_yunet.py
./track_person_xiaobai.sh             # 终端1；预览 http://127.0.0.1:8765/
./play_0001_xiaobai_tracking.sh       # 终端2；60fps、±160计数、3倍转头速度
```

此入口支持 `--check`；摄像头换 USB 口后需重启视觉服务。完整参数、重复播放与验收记录见[播放说明](PLAY_0001_XIAOBAI.md)。

人物跟踪配置和颈部方向仅经过小白验证，墨甲播放器不启用此功能；混用墨甲配置会被拒绝。Jetson 使用 Python 3.12 时，视觉依赖使用 NumPy 1.26.4。

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

## 墨甲标定转换与播放

墨甲使用 `mojia/墨甲9.23.json`，不能复用小白标定及硬件限位。52 个表情中 51 个映射发生变化；标定新增使用 ID29、35、36，停止使用 ID9、24。物理新增两轴为 ID35、36；ID29 是原有轴启用吐舌。用户提供的型号分组为 ID2–30、36：STS3032，ID31–35：SM45BL；型号字符串不等于已验证寄存器布局。

墨甲的 ID31 同时被七个表情引用。当前按用户要求使用墨甲全部映射，选择 `all_calibrated` 参与有界加权合成；零权重基准 2047 已由用户确认。小白默认保留 `JawOpen` 独占。眨眼等端点方向按墨甲文件计算，眼球沿用 30 度艺术映射，不代表实测眼球角度。

```bash
.venv/bin/python tools/prepare_mojia.py --baseline 2047 --jaw-policy all_calibrated
```

上述命令只生成离线提案：`mojia/0001_MOJIA_MOTOR_TARGETS_60fps.json` 和同名 `_REVIEW.json` 差异报告。输出是电机目标矩阵，`motor_ids` 定义列顺序，行 i 的时间为 i/fps 秒；它不是原始 61 通道 blendshape 格式，也不是可直接发送的串口数据。文件包含输入与标定 SHA256，默认标记 `playback_ready=false`，端点包络不等于机械安全限位，仍需核对实机型号、设备限位和运行限速后接入播放器。转换器不会打开串口或使能电机。媒体和生成矩阵留在本机。

传入完整 ID2–36 扫描文件 `--scan 路径` 后，转换器验证型号分组（STS3032 读数 521、SM45BL 读数 10760），将标定包络与各轴设备限位取交集，并输出同名 `_PROFILE.json`；保留全部标定映射和原始 61 通道数据。专用播放器接受生成的电机目标矩阵，核对目标、音频与标定哈希，再做实时启动检查。默认反馈故障停止；不沿用小白现场授权的反馈只记录模式。目标范围仍为标定包络与设备限位的交集；面部实际反馈单独按设备限位检查，颈部保持范围继续单独检查。ID31 首帧准备容差按本机观察设为 32 计数（其它轴默认 20），不扩大任何目标或设备位置范围。

```bash
.venv/bin/python tools/mojia_player.py check --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --output outputs/mojia/check_新编号
# 仅在现场支撑墨甲颈部、所有轴处于设备允许范围后使用；有任何轴不满足时，在首次写入前退出：
.venv/bin/python tools/mojia_player.py arm --supported-neck --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --output outputs/mojia/arm_新编号
.venv/bin/python tools/mojia_player.py play --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --output outputs/mojia/play_新编号
```

必要时用 `--port` 指定经核对的当前串口。播放器不发送 ID9、24 及颈部歌曲动画；ID32–34 保持 2047，仅监测。`outputs/mojia/STOP` 或 Ctrl+C 停止发送并保留最后目标及颈部扭矩。新建输出目录，避免覆盖日志。设备限位或标定改变后需重新生成目标和配置。

2026-10-22 实机结果：用户确认未使能时偏移正常、头部可靠支撑且无卡滞后，完成 ID29 两次观察试验（最终回到 2047），并完成 ID35、36 及颈部启动恢复。墨甲整段播放成功：112.893 秒、60fps、100% 幅度，反馈故障停止模式，发送 6773 帧、跳过 1 帧；27314 次串口读取中未记录损坏回复、读取失败或反馈通知。最高采样温度 48℃，不代表未采样时刻峰值。现场操作员已确认声音和面部动作正常；后续嘴部五轴 900 对照亦确认平顺、无明显拉扯。证据位于本机 `outputs/mojia/full_play_9/`。

实际使用文件为 `mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json` 及同名 `_PROFILE.json`；早期提案和 `DEVICE_LIMITS`/`RUNTIME` 文件用于保留调试证据，不应替代最终配置。目标文件的 `playback_ready=false` 表示仍须每次实时预检，并非绕过启动检查的运行授权。恢复与播放日志、媒体及生成目标矩阵均留在本机。

后续 ID31 动态辨识按[电机动态响应测试计划 V1.0](MOJIA_MOTOR_DYNAMIC_RESPONSE_TEST_PLAN_V1.0.md)执行。先完成 SM45BL 固件2.53的加速度语义核验，再实施单轴高频采样和参数对照。

## 墨甲右眼相机跟踪（方向待现场验证）

配置为 `mojia/person_gaze.json`，标定哈希绑定墨甲，使用独立UDP端口14787。当前只有一台HHWei USB Camera，video0为已验证图像接口；video1不能直接当作另一只眼。用户已确认当前接入右眼相机，随ID23眼球转动；配置为 `camera_mount=eye`、`camera_eye=right`、`mount_confirmed=true`。眼球及颈轴实际方向仍须独立现场验证。固定头部相机使用位置型图像误差控制，随右眼转动的相机使用图像误差积分追踪，不能直接沿用小白固定相机配置。

```bash
# 摄像头服务：只采图、检测和预览，不打开舵机串口。
.venv/bin/python tools/xiaobai_track_person.py --config mojia/person_gaze.json
# 先完成相机归属/安装关系确认，再以独立新输出目录验证小幅方向。
.venv/bin/python tools/mojia_track_person.py --config mojia/person_gaze.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --direction-trial --seconds 20 --output outputs/mojia/eye_direction_新编号
# 眼球方向正确、颈部可靠支撑后，再增加小幅ID32跟随。
.venv/bin/python tools/mojia_track_person.py --config mojia/person_gaze.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --direction-trial --neck --seconds 20 --output outputs/mojia/eye_neck_direction_新编号
```

预览为 `http://127.0.0.1:8765/`。启用 `vertical_enabled` 后追加ID7/22纵向跟踪：当前限速100、行程2027–2067，ID7抬眼时增大、ID22抬眼时减小；纵向方向仍待现场确认，当前使用 `--direction-trial`。不加 `--neck` 的当前演示仅命令8/23/7/22，颈部保留原目标且不发位置指令。横向仍为限速300、±60，未改变其已认可参数。

运动入口仅发送已配置眼轴8/23/7/22和可选颈部ID32。当前横向300计数/秒、±60；纵向100计数/秒、±20；颈部速度上限900、行程±80、软件加减速64计数/秒²。ID32最新读回硬件加速度原始64、P/D/I=16/26/0、正反死区5/5。启动核对型号、设备范围、使能、速度单位和温度；从允许范围内的当前位置接续，不自动使能或改硬件参数。不得与播放或DMS同时占用串口。

颈部采用`visual_error_assist`，不再有`eye_lead_s`计时或等待眼球跨越正中的条件。眼球控制保持原样，先响应图像误差；连续新相机帧中误差仍在增长，并且ID23反馈确认近期向该方向有位移时，颈部参与。为避免噪声，增长证据至少累计2图像像素；旧0.14/0.07数值保留，但改为归一化**图像误差**的启停门槛，不再表示眼球偏移。眼球向误差方向到达行程端点时可直接请求颈部接替；跟踪误差收敛后逐渐退出，目标反向时先制动，再按新图像方向参与，不必等待眼球归中。

保留接力增益120、滤波时间常数120ms及软件加减速约束。没有固定的眼头启动间隔；摄像头、反馈采样、滤波、轨迹加速和硬件响应仍会影响可见动作时间。颈部小目标更新条件仍为至少2计数且距上次发送至少40ms，此处是通信更新节流，不是眼球先行计时；速度命令固定900。右眼相机的真实图像闭环负责眼头补偿，没有用颈部命令假冒实际头部运动；尚未标定完整视角/传动几何，不能宣称已复现人的协调或达到最优自然度。

丢脸时眼球缓释回中、颈部减速保持；正常结束最后5秒回中，Ctrl+C或STOP文件则停止发送并保持最后目标及颈部扭矩。方向尚未确认时使用`--direction-trial`。视觉服务不占舵机串口，预览继续在`http://127.0.0.1:8765/`提供图像与控制状态。详细依据见[延迟核查与后续协调修改](docs/MOJIA_NECK_FOLLOW_DELAY_REVIEW_2026-10-23.md)。

2026-10-23：Orin上OpenCV4.10.0.84与固定YuNet模型通过10秒实拍检测，134帧均检出人脸，平均12.99Hz、处理P50为65.23ms、主机图像年龄P95为180.32ms；未连接舵机控制。软件129项测试中128通过、1项可选模型跳过，不能据此宣称墨甲眼头跟随已实机验收。

## 可选 A2F 与视频环境

`tools/requirements-env.txt` 保存完整 Python 环境版本；CUDA 12.9、TensorRT 10.13.3.9、Blender 4.5.3 和模型另行安装，二进制不入库。SDK/驱动分别通过 `bootstrap_sources.py --component a2f`、`--component driver` 恢复；驱动需针对当前内核编译，恢复命令不安装或切换驱动。

模型清单固定下载版本，例如：

```bash
.venv/bin/python tools/download_model_variant.py nvidia/Audio2Face-3D-v3.0 a2f-v3.0
source tools/env.sh
bash tools/build_a2f.sh
```

构建前需按 SDK 文档恢复其自身依赖并准备本仓库约定的本地 CUDA/TRT 目录；`install_cuda_local.py`、`install_trt_local.py` 和 `prepare_trt_layout.py` 保留本机安装辅助逻辑。歌曲编排脚本包含当前歌曲的时间点和素材路径，重跑需恢复中间产物或重新生成。此次审查未验证从空机器一键重建完整 GPU/渲染环境。

当前眼球横向300/±60、纵向100/±20已获现场认可；后续颈部调试保留这些参数。最新协调算法和ID32参数以上文为准，现场自然度及抖动效果仍需观察。

最新ID32死区对照：按现场授权写入26/27各2、46=900、41=0并读回；neck.adaptive_speed=false使速度保持900。微步合并和软件加减速200不变，眼球参数固定。锁55仍1，掉电后必须复核这些实际读数；当前试验尚待现场验收。


## 墨甲完整歌曲与实时颈眼协同

当前已保存的协同配置为`mojia/person_gaze.json`，硬件参数快照为`mojia/accepted_tracking_preset.json`；快照包含配置哈希及35轴寄存器读回，不是批量使能或写EEPROM的指令。配置调整后需重新核验并更新快照。

`tools/mojia_performance.py`统一拥有串口，将完整音频/表情与实时眼头协调合并：相机接管8/23/7/22/32，嘴部、眉毛及眼睑等保留歌曲目标；33/34保持已有姿态，9/24无墨甲动作映射且未使能，不发送虚构动作。旧小白跟踪器仍被禁止用于墨甲；仅经过墨甲校验的适配器可接入共享音频播放器。ID23至多约100ms优先轮询一次，替换常规轮询槽位，其余轴按最旧反馈优先，保留1秒新鲜度检查。实际间隔仍受控制周期和串口耗时影响。

先运行视觉服务，确保没有其它运动进程占用串口，然后执行：

```bash
.venv/bin/python tools/mojia_performance.py --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json --authorized-tongue-observation --repeat 3 --output outputs/mojia/完整演示_新编号
```

加`--check`只验证离线配置，不打开设备。`--authorized-tongue-observation`对应本次已有的ID29观察授权：仅该轴软件负载阈值1023、温度上限80℃、首帧容差40计数，保留硬件报警；如果该轴未使能，沿用已验证的2047目标、速度900启动流程，10秒内必须进入40计数走廊，最终仍按设备范围检查。全部演示结束或中止后单独卸载ID29，颈部保持使能。其余轴没有该例外。

每遍最后5秒仅让跟踪轴平滑回中，音乐和其余表情完整播放；下一遍从实测眼头姿态接续。初始过渡不以900速度强制颈部归中。每遍生成独立`playback_log.json`，总结果在`summary.json`；预览保持`http://127.0.0.1:8765/`。
