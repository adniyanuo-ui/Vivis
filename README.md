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

配置为 `mojia/person_gaze.json`，标定哈希绑定墨甲，使用独立UDP端口14787。当前接入一台HHWei USB Camera，使用配置中的by-id图像接口；video节点编号可能随插拔改变，不能据此判断另一只眼。用户已确认当前接入右眼相机，随ID23眼球转动；配置为 `camera_mount=eye`、`camera_eye=right`、`mount_confirmed=true`，右眼整曲联合效果已获现场认可。更换相机侧或机构后仍须验证实际方向。固定头部相机使用位置型图像误差控制，随眼球转动的相机使用图像误差积分追踪，不能直接沿用小白固定相机配置。

```bash
# 摄像头服务：只采图、检测和预览，不打开舵机串口。
.venv/bin/python tools/xiaobai_track_person.py --config mojia/person_gaze.json
# 先完成相机归属/安装关系确认，再以独立新输出目录验证小幅方向。
.venv/bin/python tools/mojia_track_person.py --config mojia/person_gaze.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --direction-trial --seconds 20 --output outputs/mojia/eye_direction_新编号
# 眼球方向正确、颈部可靠支撑后，再增加小幅ID32跟随。
.venv/bin/python tools/mojia_track_person.py --config mojia/person_gaze.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --direction-trial --neck --seconds 20 --output outputs/mojia/eye_neck_direction_新编号
```

预览为 `http://127.0.0.1:8765/`。启用 `vertical_enabled` 后追加ID7/22纵向跟踪：当前限速900、行程2027–2067，ID7抬眼时增大、ID22抬眼时减小；方向已由现场确认。不加 `--neck` 的演示仅命令8/23/7/22，颈部保留原目标且不发位置指令。横向限速也为900，行程仍为±60。

运动入口仅发送已配置眼轴8/23/7/22和可选颈部ID32。当前横向900计数/秒、±60；纵向900计数/秒、±20；颈部速度上限900，按2026-10-24用户要求使用设备完整位置范围（最近读回1600–2600），已移除软件加减速限制。ID32最新读回硬件加速度原始0、P/D/I=16/26/0、正反死区5/5。启动核对型号、设备范围、使能、速度单位和温度；从允许范围内的当前位置接续，不自动使能或改硬件参数。不得与播放或DMS同时占用串口。

颈部采用`visual_error_assist`，不再有`eye_lead_s`计时或等待眼球跨越正中的条件。眼球控制保持原样，先响应图像误差；连续新相机帧中误差仍在增长，并且ID23反馈确认近期向该方向有位移时，颈部参与。为避免噪声，增长证据至少累计2图像像素；旧0.14/0.07数值保留，但改为归一化**图像误差**的启停门槛，不再表示眼球偏移。眼球向误差方向到达行程端点时可直接请求颈部接替；跟踪误差收敛后逐渐退出，目标反向时先制动，再按新图像方向参与，不必等待眼球归中。

保留接力增益120及滤波时间常数120ms；颈部软件加减速约束已按用户要求移除。没有固定的眼头启动间隔；摄像头、反馈采样、滤波、轨迹加速和硬件响应仍会影响可见动作时间。颈部小目标更新条件仍为至少2计数且距上次发送至少40ms，此处是通信更新节流，不是眼球先行计时；速度命令固定900。右眼相机的真实图像闭环负责眼头补偿，没有用颈部命令假冒实际头部运动；尚未标定完整视角/传动几何，不能宣称已复现人的协调或达到最优自然度。

丢脸时眼球缓释回中、颈部停止推进目标并保持；正常结束前按回中距离/速度上限加5秒预留时间窗。Ctrl+C或STOP文件则停止发送并保持最后目标及颈部扭矩。方向尚未确认时使用`--direction-trial`。视觉服务不占舵机串口，预览继续在`http://127.0.0.1:8765/`提供图像与控制状态。详细依据见[延迟核查与后续协调修改](docs/MOJIA_NECK_FOLLOW_DELAY_REVIEW_2026-10-23.md)。

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

历史现场认可的眼球参数为横向300/±60、纵向100/±20；2026-10-24按用户要求将两者限速统一为900，行程和协调算法保留，速度调整后的现场效果尚待观察。最新ID32参数以上文为准。

历史ID32死区对照（已被后续调参替代）：当时写入26/27各2、46=900、41=0，软件加减速200。当前有效值见上文和`mojia/person_gaze.json`，不要把本段历史记录当作当前参数。


## 墨甲完整歌曲与实时颈眼协同

当前已保存的协同配置为`mojia/person_gaze.json`，硬件参数快照为`mojia/accepted_tracking_preset.json`；快照包含配置哈希及35轴寄存器读回，不是批量使能或写EEPROM的指令。配置调整后需重新核验并更新快照。

`tools/mojia_performance.py`统一拥有串口，将完整音频/表情与实时眼头协调合并：相机接管8/23/7/22/32，嘴部、眉毛及眼睑等保留歌曲目标；33/34保持已有姿态，9/24无墨甲动作映射且未使能，不发送虚构动作。旧小白跟踪器仍被禁止用于墨甲；仅经过墨甲校验的适配器可接入共享音频播放器。

联合模式按单调时钟60 Hz定时发送，音频DAC时钟选择当前歌曲帧；错过周期后直接取最新目标，不补发过期帧。位置发完后，反馈只占用本周期剩余时间：最多7ms、8个短请求，每个请求3ms超时；不在控制周期内等待10ms并立即重试。四段反馈可跨周期完成，缺失或校验错误会丢弃整份未完成样本，后续重新采集，不更新有效反馈时间。ID23/32超过80ms未刷新时优先，其余轴按最旧有效反馈优先。80ms是调度阈值，实际刷新受读取和系统调度影响。全部轴1秒新鲜度、通信失败比例5%、温度/负载/状态及100ms控制看门狗继续生效。真实传感器异常仍走原有暂停发送、确认或停止流程，不能为顺滑而忽略报警。

音乐开始前的起步/到位检查如遇连续坏包，暂停发送新位置，间隔20ms重新采集完整反馈，最多4次新采样、恢复窗口0.5秒且不超过原启动期限；连续两次有效并通过原保护检查才继续。真实报警和通信失败比例超限不走此恢复。恢复结束重置起步计时时刻，防止把暂停时间累计成下一步的大位移。日志记录`phase`和`startup_communication_recoveries`；该恢复不用于音乐实时段。

颈部的软件与硬件加速度分开处理：原`neck.acceleration_counts_s2=64`已按用户要求移除，跟踪及播放共用控制器不再使用软件速度斜坡或基于该加速度的限位前制动。`neck.observed_acceleration_raw=0`仍记录ID32硬件地址41的读回值。视觉协调请求直接用于位置积分，并保留速度上限和位置限位；丢脸/撤销请求后不再累积旧速度。硬件加速度原始0与软件移除限制是两项独立设置。

先运行视觉服务，确保没有其它运动进程占用串口，然后执行：

```bash
.venv/bin/python tools/mojia_performance.py --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json --authorized-tongue-observation --usb-volume-times-three --repeat 3 --output outputs/mojia/完整演示_新编号
```

加`--check`只验证离线配置，不打开设备。`--authorized-tongue-observation`对应已有的ID29观察授权：仅该轴软件负载阈值1023、温度上限80℃、首帧容差40计数，保留硬件报警；如果该轴未使能，以2047目标、至多本次回中速度（默认60）进行既有恢复，10秒内必须进入40计数走廊，最终仍按设备范围检查。全部演示结束或中止后单独卸载ID29，颈部保持使能。其余轴没有该例外；用户层特殊设置清理需等待用户确认。

每遍尾段仅让跟踪轴平滑回中，预留时间至少5秒，音乐和其余表情完整播放；下一遍从实测眼头姿态接续。初始过渡不以900速度强制颈部归中。每遍生成独立`playback_log.json`，总结果在`summary.json`；预览保持`http://127.0.0.1:8765/`。

完整颈部范围下，回中时间窗按当前规划位置至2047的距离/有效速度上限加5秒估计，进入回中后锁定该阶段；保留接近中位时的比例速度收敛，不使用加速度估计。`neck.range_mode=device`从硬件profile读取两侧不同限位，不再使用原±80。真实启动读回仍必须与profile设备范围一致；到达端点的最后1计数也允许发送。范围扩大允许更大跟随，不会主动扫到端点，也不保证小人脸偏差会产生大转角。

推荐手动入口（自动复用或启动相机预览）：

启动前确认没有系统服务抢占设备：`ies.service`中的`face_track`会占用USB相机，`dms.service`中的头/身体节点会占用串口并可能重写舵机参数。此前停止服务不代表重启后仍停止；需要本次移交给Vivis时先执行`sudo systemctl stop ies.service dms.service`。手动入口检测到它们仍活动会明确退出，不自动关闭系统服务。串口重新插拔后如提示权限不足，可执行`sudo setfacl -m u:$(id -un):rw /dev/ttyCH343USB0`；不要用root运行整个播放器。仅停止DMS不会恢复已被它覆盖的舵机设置，仍需核对已确认的硬件参数。

```bash
cd /home/james/project/Vivis
./play_mojia_full_show.sh --repeat 1
```

默认完整播放三遍；仅检查用`./play_mojia_full_show.sh --check`，不会启动相机、修改音量或打开舵机串口。前台运行可用Ctrl+C停止。每次生成独立目录，路径记录在`outputs/mojia/latest_full_show.txt`。STOP文件存在时拒绝开始，不会自动清除。

2026-10-24新增执行前35轴回中：用户确认ID2–36（含9/24）均以2047为中位，设备清单见`mojia/homing_axes.json`。默认临时速度上限60计数/秒，先暂停跟踪/音乐，用公共五次平滑曲线回中，完成后恢复各轴原速度再开始任务，不写PID、死区或新的硬件加速度。曲线起止的命令速度/加速度为零；实际机械顺滑度待观察。9/24不参与歌曲，回中时临时使能，完成后恢复原使能状态；7/22在回中后继续保持原任务中的暂停状态。中止时保留低速及最后目标，不突然恢复高速。

先只观察回中：`./play_mojia_full_show.sh --home-only`，不启动相机、音频或音量设置。调整速度用`--home-speed 40`或`--home-speed 80`。随后执行`./play_mojia_full_show.sh --repeat 1`，会再次先回中再播放。日志`summary.json`的`homing`记录全部中位、起始实测、回中速度、到位及原速度恢复结果，`home_inventory.json`保留本次设备清单，`homing_serial_trace.json`记录启动/回中串口数据。当前回中仍继承原到位容差和保护策略，这些列在待确认清单，尚未删除。

用户层硬件相关设置清单共32项，见[代码审查记录](docs/CODE_REVIEW.md)中“2026-10-24 35轴回中与用户层硬件参数待确认清单”；包括当前生效项、协同算法项、设备标定、历史路径和通信策略，逐项区分实际写入、校验、限步和历史证据。

2026-10-24已完成左眼相机切换试验，随后按用户确认换回右眼，当前`mojia/person_gaze.json`的`camera_eye=right`。两只相机曾报告相同HHW000标识，不能仅靠by-id判断物理左右；当前选择依据现场确认且只接一只。软件根据相机侧选择参考轴：左眼8/7，右眼23/22；编码器运动证据和联合播放优先反馈随之切换。已认可的速度、PID及协同参数保留，ID32行程采用上述设备范围。单独跟踪增加`--hold-vertical-eyes`，与播放入口一致地暂停7/22；预览仍在127.0.0.1:8765，共享状态目录的历史名称不代表当前相机侧。阶段基线af1e7b0保留此前配置及硬件快照。

回中反馈使用设备行程，任务开始后使用配置的跟踪行程；例如旧±80配置下，当前姿态在跟踪范围之外、但在设备范围内时也能缓慢回中。若编码器只在中位附近存在已有到位容差内的残差（如ID9实测2052而设备目标上限2050、中位2047），回中按原到位条件验证；所有下发目标仍夹在设备命令范围内，不为该轴添加新的专用容差。回中结束另外读取35轴目标RAM确认中位，速度/使能写入如果丢失应答，使用两次独立读回确认，不重发写入；真实报警或读回不符仍停止。

整合验收先完整播放一遍，现场有人脸持续出现在画面中，分别观察左右跟随、反向及停住；通过后用`--repeat 3`连续观察。播放后运行`.venv/bin/python tools/analyze_mojia_show.py`，自动读取最近一轮目录并生成`performance/timing_report.json`，检查指令间隔、ID23/32反馈间隔、通信失败和UART帧错误。报告中的位置误差来自分时采样，不等于高频机械振动测量。

对照30 FPS面部采样可用`./play_mojia_full_show.sh --repeat 1 --face-fps 30`。音乐时长和眼颈60 Hz控制保持；只按30 Hz抽取歌曲表情目标，零阶保持至下一样本，可能丢失短促口型峰值。该选项减少歌曲目标变化次数，不承诺总线流量减半或机械效果更好。默认仍为60 FPS。不要直接把整个控制循环改成30 Hz且维持每帧一轴轮询，否则33轴和优先反馈会超过原有1秒新鲜度期限。

2026-10-24用户认可口型张开及颈部大范围跟随效果，手动入口已保存默认口型增益1.5：`./play_mojia_full_show.sh --repeat 2`即可复现两遍。只对12/13/27/28/29/31应用`neutral + gain × (original-neutral)`，再截断至原标定/设备范围交集；眼颈、其它表情、声音幅度与速度900不因该增益改变。对照原始幅度用`--mouth-gain 1.0`覆盖；底层Python工具的通用默认仍为1.0，原目标文件不改写。每次输出`mouth_gain_report.json`记录请求/实际范围和被限幅帧数，增益1.5不代表实际机械幅度在所有帧都能达到1.5倍。当前验收配置哈希、硬件读回及手动入口选项记录在`mojia/accepted_tracking_preset.json`；动态坏包约0.79%的问题按用户要求后续处理，原通信检查保留。

三倍音量按输出信号幅度定义，以此前USB Speaker第15档（−22.5dB）为固定基准：启动时设置第23档（−10.5dB，+12dB），数字衰减系数0.7535659295，总幅度约为原来的3倍。每次设置相同档位，不对当前音量累乘；原始WAV与动作文件不修改。已对实际48kHz重采样音频检查峰值低于1，避免直接数字乘3导致削波。设备型号、档位范围或读回dB不符时停止，不猜测其它设备的音量范围。

2026-10-24当前眨眼对照使用0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1200.json及同名_PROFILE.json。1800试验在44.65秒因ID5瞬时负载384超过既有350门槛中止；低速回中卸除闭眼负载后，按用户反馈改为1200、加速度保持0、位置限位不变。1800与1200目标位置矩阵、FPS、帧数和眨眼阶段帧索引完全相同，不因降速延长动作；允许闭合幅度变浅。频次从26次减到18次（约减少31%，本片约9.57次/分钟）。按局部音频强弱、表情重要性及相邻间隔确定性删选，保留表达性长闭眼及首尾事件，不固定每隔几次删一次。删除事件会从源眨眼权重中移除，并复用原映射重合成四个眼睑列，避免只取消增强、原眨眼实际仍在的错误。其它表情输入、口型1.5倍与眼颈跟踪参数不变。

普通自然眨眼的参考时序为闭合约100ms、张开约220ms，总长约320ms，见[原始研究摘要](https://pubmed.ncbi.nlm.nih.gov/19157026/)。本片14次普通事件的目标闭合阶段83–100ms、张开200–233ms、平台0–17ms，总长283–350ms、均值315ms；另4次保留原表达性半闭眼/长闭眼变化。前置正弦闭合与缓起的张开曲线是参照自然眨眼的工程调度，不声称复制生理肌肉或测得真机320ms。此前1800离线限步仿真中普通事件左上眼睑峰值行程为94.0–99.1%、中位96.6%，该数据不代表当前1200下的表现；真实闭合幅度和机械持续时间仍需现场观察。原标定闭眼目标仍为5=2320、6=2150、20=1803、21=1943，端点未扩大。

旧BLINK900（统一长平台）与BLINK_CONTEXT（900、按音频表情变时长）文件保留作对照。当前生成器默认1200及保留比例0.7，生成最新方案；当前没有核验的歌词逐句时间轴，混合音频强弱只作为声学表演线索，不代表准确识别歌词句界。用户已确认1200整曲全部正常，当前profile与accepted_tracking_preset.json已更新验收记录和35轴只读快照。

该目标文件按既有资源策略留在本机；重新生成用以下离线命令，输入输出均不打开串口，不覆盖已有版本：

```bash
.venv/bin/python tools/adjust_mojia_blinks.py \
  --targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json \
  --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json \
  --clip exmple/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json \
  --speed 1200 --keep-fraction .7 \
  --output mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1200.json
```

恢复上一版使用手动入口附加参数：--targets mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK.json --profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_PLAYBACK_PROFILE.json。该参数覆盖仅切换目标与运行profile；若已写入眼睑速度RAM，需另行核对，播放时按所选profile发送速度。profile中的blink_trial保留本版验收状态，不替代完整启动检查。

实时段的预览状态写盘移至独立线程，仅保留最新状态；写盘阻塞不会等待在主控制线程中。循环垃圾回收移出音频实时段，广播后保留1ms额外静默和UART线时。日志记录每帧视觉计算、指令发送、反馈三个阶段耗时，以及分步读取次数和完整反馈样本。2026-10-24整曲及随后连续两遍完成，现场认可口型和颈部大范围跟随；定时调度并非硬实时保证，动态通信坏包和长期稳定性仍需后续处理。

2026-10-23最新演示设置：手动入口默认追加`--hold-vertical-eyes`，ID7/22保持当前位置，不接收歌曲或相机的位置指令，仍保留扭矩和反馈检查；横向8/23及颈部32继续跟踪。纵向行程仍为±20，2026-10-24已将限速从100统一为900；完整播放仍不启用纵向动作。两遍演示可运行`./play_mojia_full_show.sh --repeat 2`。

Ctrl+C及STOP文件属于主动停止：两个完整播放入口会停止音频和位置发送，释放串口，按原策略卸载ID29并保持颈部扭矩，打印正常停止提示，不再显示异常回溯；进程退出码为130，日志保留`completed=false`、`stopped_by_operator=true`及停止原因。真实通信或舵机故障仍保留异常记录，不当作主动停止吞掉。

结束或中止后ID29已卸载，下一次启动会进入低速恢复。该路径现与35轴扫描/回中复用有界坏包恢复：损坏/缺失回复作废，最多4次新读取中必须重新取得两次连续、通过原保护检查的完整反馈；恢复窗口不超过0.5秒，并受10秒启动截止时间约束。只重读反馈，不因此重复位置或使能写入。退出时确认ID29已卸载后，恢复它的任务速度上限，避免临时60残留导致下一次900启动核对失败；未确认卸载则不提高速度。持续坏包、读取失败率门槛、真实报警、负载/温度异常仍停止，底层通信坏包成因仍待后续处理。

### 墨甲当前统一速度配置（2026-10-24）

当前两个作品的现用profile及眼颈跟踪配置均为：已映射的33轴中，ID5/6/20/21四个眼睑保留1200计数/秒，其余29轴统一900；未映射、未使能的ID9/24不改，读回速度仍为0。已写入速度RAM46并读回35轴，记录在`mojia/accepted_tracking_preset.json`。这是任务速度上限，轨迹规划及图像误差仍会影响实际速度；不代表每帧都以900运动。ID7/22在完整播放中仍保持回中位置，ID33/34仍不参与位置跟踪。

最新加速度RAM41：2026-10-24按用户追加要求，将ID32从64、ID33/34/35从10改为0，35轴读回均为0。此前33/34/35的10为历史硬件值，最初设置来源不明。0在当前SM45BL固件中的有效物理加速度仍待核实，不能据原始值宣称已测得最大物理加速度。随后已按用户要求移除颈部软件加减速64计数/秒²；速度、PID、死区、位置限位、负载阈值和扭矩使能均未改。

启动回中仍临时使用60计数/秒，完成后恢复任务速度。旧PLAYBACK、BLINK900、BLINK_CONTEXT、BLINK_NATURAL1800等历史对照profile保留原值；显式切换旧profile会采用它的配置，应按历史对照使用。当前速度及加速度设置已完成离线及寄存器核对；最新硬件加速度改0及软件加减速移除后的现场效果仍待观察。之前的自然度验收记录属于调整前基线`f14efec`。

### 墨甲第二作品 wz2

两个作品分别使用独立资源和入口，共用已验收的控制流程：

| 作品 | 播放命令 | 默认遍数 |
|---|---|---|
| 0001 | `./play_mojia_full_show.sh --repeat 1` | 未指定时3遍 |
| wz2 | `./play_mojia_wz2.sh` | 1遍，可加`--repeat 2` |

第二作品源文件实际位于exmple目录：wz2_BLENDSHAPES_61_30fps.json（823帧、30FPS）与wz2_CLAIRE_VISIBLE_BROWS_v3.0-Claire.mp4。视频AAC音轨解码为16kHz双声道PCM16，核对GStreamer首缓冲PTS与播放段起点均为64ms、段映射time=0，音频从媒体时间0开始。保留全部解码PCM，仅尾部补661采样（41.3125ms）静音以对齐动画，准备后音频约27.433秒。30FPS原样本逐帧保持两个60Hz周期，生成1646帧，不改变作品节奏。源视频和JSON保留原样。

沿用墨甲标定、35轴60计数/秒前置回中、眼睑1200、嘴部900、口型增益1.5、右眼相机/横向双眼及颈部跟踪，ID7/22回中后保持，ID32负载阈值350。当前眨眼规则从新作品原9次选保留6次，按新作品音频及表情安排。wz2使用独立媒体、目标JSON和profile，不覆盖0001文件。两入口均支持--check，只做离线核对。当前入口选用wz2_expression15版本：仅眉毛2/3/4/17/18/19和鼻翼14/16的合成后中位位移放大1.5倍，按原标定/设备范围截断；嘴部、眼睑、眼球、颈部目标及音频保持不变。原增益1.0版文件保留作对照。

重新准备资源（以下输出已存在时会拒绝覆盖，需另选前缀）：

```bash
gst-launch-1.0 -q filesrc location=exmple/wz2_CLAIRE_VISIBLE_BROWS_v3.0-Claire.mp4 \
  '!' qtdemux name=demux demux.audio_0 '!' queue '!' decodebin \
  '!' audioconvert '!' audio/x-raw,format=S16LE,layout=interleaved \
  '!' wavenc '!' filesink location=mojia/wz2_AUDIO_FROM_MP4_RAW.wav
.venv/bin/python tools/prepare_mojia_work.py \
  --clip exmple/wz2_BLENDSHAPES_61_30fps.json \
  --decoded-audio mojia/wz2_AUDIO_FROM_MP4_RAW.wav \
  --video exmple/wz2_CLAIRE_VISIBLE_BROWS_v3.0-Claire.mp4 \
  --reference-profile mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1200_PROFILE.json \
  --expression-gain 1.5 --prefix mojia/wz2_expression15
.venv/bin/python tools/adjust_mojia_blinks.py \
  --targets mojia/wz2_expression15_MOJIA_MOTOR_TARGETS_60fps.json \
  --profile mojia/wz2_expression15_MOJIA_MOTOR_TARGETS_60fps_PROFILE.json \
  --clip mojia/wz2_expression15_BLENDSHAPES_61_60fps_HOLD.json --audio mojia/wz2_expression15.wav \
  --speed 1200 --keep-fraction .7 \
  --output mojia/wz2_expression15_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1200.json
```

新工具仅为已核对时间原点的PCM16解码音频补不超过100ms尾部静音，音频更长需裁剪或差异更大时会拒绝，避免悄悄裁剪内容。原媒体及大型生成目标仍按既有策略留在本机，启动脚本、生成工具和profile纳入代码管理。

脸颊标定缺项：当前墨甲CheekPuff.positions为空，CheekSquint只映射下眼睑6/21，ID9/24未被任何表情引用。源JSON的CheekPuff虽有非零权重，也无法经空映射生成独立脸颊动作。需要现场确认实际脸颊ID及动作端点后补标定，不能用其它机构映射代替；本轮未给9/24增加动作，也未扩大硬件限位。
