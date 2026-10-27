# 墨甲开发者快速上手

版本0.2｜2026-10-26｜示例3已完成wz2一遍运行；新ID32抖动及其他验收项见[验收表](ACCEPTANCE.md)。

## 1. 准备环境

机电头、匹配电源、USB串口通信连接、客户自备Linux主机及音频输出。示例2/3需要一只已配置的眼内USB相机，Orin主控可选。需要Python3.10–3.12；本机为3.12。

本体包含35轴和两只眼内相机。USB控制器到TTL/RS485的转接、电平转换及供电应按实际本体线束/控制板连接，不能把不同电平端子直接互连。

## 2. 安装

仓库根目录执行。新Ubuntu主机缺少系统依赖时先安装；已有环境不必重装驱动、CUDA或系统软件。

```bash
sudo apt-get install python3-venv libportaudio2 alsa-utils build-essential cmake
./tools/setup.sh
```

脚本只建立`.venv`、用pip安装Python依赖、核验随包资源并做离线检查，不访问串口或相机，不修改系统服务、USB驱动或舵机参数。模型和SDK已随Git提供，不另行下载。普通Linux主机不要求CUDA；Windows尚未验证。

启动脚本只将本仓库`src/`加入`PYTHONPATH`，不继承全局ROS或旧工程的Python路径，并禁用用户site-packages注入。额外依赖应安装进本项目`.venv`；该处理不删除系统ROS，也不修改用户的登录环境。

手动安装等价操作：

```bash
python3 -m venv .venv
CMAKE_BUILD_PARALLEL_LEVEL=2 .venv/bin/python -m pip install -e '.[vision,trajectory]'
.venv/bin/python tools/check_assets.py
```

## 3. 素材已包含

两份作品的音频和目标轨迹在`assets/works/`，YuNet在`assets/models/yunet/`，舵机SDK子集在源码包内。清单和哈希见[素材说明](../../assets/README.md)，无需从旧目录复制文件或恢复软链接。

## 4. 配置设备

入口为`configs/robots/mojia/robot.json`。核对串口、波特率、音箱和相机节点；示例2/3可用`--camera-device /dev/v4l/by-id/实际节点`指定。串口USB重插后需仍有权限，部署规则见`tools/99-vivis-servo-access.rules`。

默认右眼参考ID23，相机640×480/30FPS、检测目标20Hz、运动60Hz，各频率相互独立。横向双眼1987–2107，颈部32跟踪指令范围1706–2388。用户已确认ID32新增EPROM限位1706–2388；设备与跟踪范围均保存在`hardware.json`，启动核验且不自动改写限位。500kbps候选未启用，默认仍1Mbps。启动器只提示占用服务，不自行停服务、重启USB或切驱动。

音频设备名保存在`hardware.json`的`runtime.audio_device`。默认音量操作针对已核验的USB Audio Device；客户改用其他音箱/声卡时，应在`robot.json`中将`defaults.usb_volume_times_three`设为`false`，配置实际输出后自行设音量。不能把本机USB音箱的固定档位套用到其他声卡。

串口权限可由管理员按实际适配器核对`tools/99-vivis-servo-access.rules`后安装；其中仅匹配当前适配器的厂商/产品ID。使用`dialout`组访问时，将用户加入该组后重新登录，USB重插仍需核验节点及权限。一次性授权可按报错中的`setfacl`命令处理。

## 5. 离线检查

```bash
./examples/01_face_control/run.sh --check
./examples/02_face_eye_tracking/run.sh --check
./examples/03_face_eye_neck_tracking/run.sh --check
# 完整离线交付检查（包含上述六组入口、素材及回归测试）：
./tools/check_delivery.sh
```

检查素材/标定哈希、目标范围、回中清单、速度及轴控制权；不打开串口，不启动音频或相机。通过不代表设备在线或机械动作已验收。

## 6. 真机验收和停止

核对本体供电、通信及头颈支撑，按三个示例README逐个运行。先wz2，再选择`--work 0001`。每遍独立准备、按需执行既有机构恢复、35轴以速度60回中，恢复任务速度后完整播放。回中会涉及35轴，“保持”指正式播放阶段。

使用[验收表](ACCEPTANCE.md)记录，之前版本的效果不直接计作本分支入口已通过。日志为`outputs/developer/时间_模式/`，`performance/summary.json`记录总体结果，`round_N/preparation.json`和`playback_log.json`记录准备和播放；相机日志为`vision.log`及`vision/summary.json`。

Ctrl+C停止发送和音频，释放接收器、串口及自行启动的相机，保留颈部扭矩，沿用既有舌部卸载策略。舵机可能继续完成最后一个目标；停止发送不等于断电急停。作品结束不按静音停顿改写口型，也不强制面部回中。

## 7. 冷启动时未使能轴的小幅偏移

上电后舵机可能尚未使能，反馈位置会因被动机构位置超出设备目标范围。当前`homing.json`配置允许未使能轴最多10计数的启动偏移，连续复读2次且相差不超过2计数后，先以不超过60计数/秒的速度预置限位内目标，读回位置目标/速度，再使能并缓慢回中。没有修改设备限位、标定或播放位置范围。

发送启动目标前核验使能状态为0；本机部分轴在收到合法位置更新后会自行使能，因此目标发出后接受状态为1，仍为0才显式使能。两种路径都读回目标、速度和最终使能状态，记录在回中日志的`startup_primes`中；目标发送前的意外使能仍拒绝继续。ID9/24只临时参加回中，结束后恢复卸载。

恢复时限覆盖共同规划的缓慢回中轨迹，再加配置中的恢复余量，并受总回中超时约束。测量进入设备范围即可结束恢复；若仍有边缘编码器残差，仅在已发送并读回中位目标后，允许沿用既有回中容差，记录在`startup_neutral_residuals`。播放阶段的位置检查保持原范围。

恢复只在回中阶段生效；进入设备范围后撤销临时反馈通道，仍在范围外超过配置时间或最终未回中则停止。已使能轴、无效编码器、异常状态和超出允许幅度的偏移不按此规则放行。本机ID35经现场确认机构正常，其未使能被动偏移单独配置为最多160计数、重入等待5秒，速度仍不超过60；ID29继续沿用既有下颌让位/收舌流程。更大的已知被动行程必须逐轴确认、显式配置，不能通过扩大全部轴阈值处理。
