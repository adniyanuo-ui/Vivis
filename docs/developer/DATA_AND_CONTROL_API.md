# 数据、映射与控制接口

版本0.2｜2026-10-26。本包提供Python源码接口和命令行示例；监控HTTP服务不是表情控制REST API。C/C++/Java客户端、ROS2及实时流式A2F接口未作为本轮已实现功能。

## 1. 输入约定

示例使用PCM16 WAV及派生电机轨迹，`assets/works.json`记录来源哈希。上游原始61通道文件格式如下，`blendShapes`中的每一行必须有61个数：

```json
{"name":"my_clip","fps":{"num":60,"den":1},"blendShapes":[["此处替换为61个数值"]]}
```

上面数组内容只是形状说明，不是可执行样例。真实名称与顺序见`configs/schemas/blendshape_channels_61.csv`：前52维为0–1基础表情，后9维为头部及双眼旋转，沿用导出的弧度约定。未实现通道应明确为0或空映射，不能错位填充。

若客户上游使用具名表情、`timestamp_ms`及独立的注视/头姿字段，需要显式转换到本例文件约定。当前离线输入以帧率定位时间，不提供通用表情网络协议；LLM情绪标签也不能直接当表情权重。

## 2. 可直接调用的纯映射API

在仓库editable安装后运行：

```python
from vivis.runtime.robot import Robot
from vivis.motion.expression import ExpressionMapper, adapt_channels, fuse_layers

robot = Robot()
mapper = ExpressionMapper(robot)
# 未提供的基础表情为0；不打开串口、不驱动设备。
base = adapt_channels({"brow_down_left": 0.5},
                      {"brow_down_left": "BrowDownLeft"}, mapper.names)
weights = fuse_layers(base, [({"BrowDownRight": 0.3}, 1.0)], mapper.names)
nominal_positions = mapper.frame(weights)
print(nominal_positions)
```

`adapt_channels`只接受显式名称映射，拒绝未知通道、重复目标和非有限/越界权重。它不解决不同数字角色基底的语义差异，客户应先确认表情定义。

`fuse_layers`仅覆盖层中明确出现的通道，强度0–1，按`旧值×(1−强度)+新值×强度`顺序融合。后层优先，是供客户显式使用的接口；两份已有作品默认不经过新增表情叠加。

`mapper.frame`返回30个当前标定轴的浮点名义位置。共享轴和眼睑使用现有合成规则，空输入返回中位；不进行音频静音识别。返回值尚未经过完整运行时的位置、速度和反馈检查，也不是已经发送的目标。运动加减速由舵机SRAM41控制，不再叠加软件固定加减速限制。

## 3. 接入自己的音频与表情序列

先确保PCM16 WAV与61通道文件时间长度一致。导入命令不连接设备、不修改输入，会使用当前硬件配置的已保存范围生成派生文件，添加一个新作品条目：

```bash
PYTHONPATH=src .venv/bin/python -m vivis.motion.export \
  --name my_demo --clip /path/to/clip.json --audio /path/to/audio.wav
./examples/01_face_control/run.sh --work my_demo --check
./examples/01_face_control/run.sh --work my_demo
```

当前接受30或60FPS源文件。30FPS逐帧保持两次形成60FPS派生轨迹，音频及原文件保持不变；不插入表情事件，不根据静音修改目标。新文件写入`outputs/imported_works/my_demo/`，清单写入`assets/works.json`。重名拒绝覆盖；提交或交付新作品时同时携带其素材，不能只交清单。生成使用保存的限位，不代表完成了新素材的真机验收。

三个示例`main.py`展示如何选择行为组合；真正会话实现为`runtime/developer.py`与`runtime/show.py`。控制核心负责启动设置、回中、音频时钟、统一目标发送和反馈，不建议把三个示例改成各自直接操作串口。

## 4. 唯一配置来源

| 文件 | 唯一负责的数据 |
|---|---|
| `robot.json` | 其他配置引用、素材清单、相机选择及示例默认值 |
| `calibration.json` | 最终52基础表情到轴端点的关系 |
| `hardware.json` | 完整35轴清单、型号、中位、设备/软件范围、速度、SRAM加速度与硬件调参值 |
| `tracking.json` | 图像误差控制、眼球端点、滤波、丢脸和协调参数，不再保存第二套速度/PID |
| `homing.json` | 回中速度、未使能位置复核与逐轴被动偏移恢复规则 |
| `telemetry.json` | 软件负载/温度告警和连续高温停止策略 |
| `assets/works.json` | 音频及派生轨迹的路径和来源哈希，无硬件配置副本 |

`Robot`按ID生成运行profile、35轴回中清单、启动寄存器设置和跟踪观察参数；运行日志保存其有效值。作品和运行参数分离，不保留另一套作品profile或历史配置。

设备寄存器范围与任务运动范围分别保存。ID32在2026-10-26未使能启动状态连续三次读回设备范围0–4095，`device_min/max`据此核验设备；已接受的跟踪范围1600–2600保存在同轴的`tracking_min/max`。`tracking.json`选择`range_mode="configured"`后从硬件配置取这两个端点，不把设备全编码器范围当作实际机构行程。启动检查仍要求设备限位与保存值一致，不自动改写限位寄存器。

改变速度以`hardware.json`为准；实际启动会读回核验。改变标定需重新生成受影响轨迹并核对其来源哈希，不能仅修改哈希使旧轨迹通过校验。波特率修改必须主机和35轴一致，当前示例不会自动写设备波特率。

颈部目标平滑时间常数`neck.target_filter_tau_s`和回中参考最短时长`neck.return_duration_s`在`tracking.json`中定义，分别为0.15秒、3秒；属于行为目标的连续化处理，不恢复固定64计数/秒²软件加速度限制。`person_gaze`日志分别保存`planned_neck_goal`（平滑前）、`smoothed_neck_goal`（平滑后）和`neck_return_duration_s`；`events.goals`是交给发送器的整数请求，须结合`transmitted_ids`确认该周期实际发送的轴。这些均不等于编码器位置。当前32号SRAM41=128仍按唯一硬件配置恢复。

16计数发送死区已因现场分段运动撤销，相关配置、SDK选项及运行日志字段已移除；旧日志中的`goal_deadband`仅属于当时的试验版本。ID32 PID从唯一硬件配置取得16/32/0。

[上游轨迹规划](UPSTREAM_TRAJECTORY_PLANNING.md)默认关闭，由`neck.upstream_trajectory.enabled`或示例3的`--upstream-trajectory on/off`控制。开启时用单轴jerk受限规划接管ID32的位置低通，`planned_neck_goal`/`smoothed_neck_goal`均为规划输出，不表示再次滤波；原始期望速度和参考p/v/a、平均jerk、模式、周期在`person_gaze.upstream_trajectory`记录。关闭时保持150ms链路和原输出，且不导入可选Ruckig依赖。开关只在启动时解析，不支持运行中热切换。

`neck.eye_assist_ratio=0.25`定义当前相机眼的实测行程使用率辅助门槛，不是固定等待时间。`person_gaze.eye_offset_command`与`eye_offset_measured`分别记录命令和实测归一化眼位，`eye_feedback_age_s`记录反馈年龄；`eye_reserve`表示图像仍有偏差且已消耗眼部行程，`eye_feedback_stale`表示反馈过期而不请求颈部辅助。没有新增串口轮询，复用现有眼球反馈。

`--mouth-gain`沿用既有6轴偏移倍率，适用ID12/13/27/28/29/31，在`hardware.json`的`runtime.mouth_gain_ids`中列明；它与14个口型相关轴的速度分组是两个概念。默认倍率1.0，不改变目标幅度。

## 5. 控制权与资源

| 模式 | 播放发送轴数 | 跟踪拥有轴 | 播放保持 |
|---|---:|---|---|
| face | 26 | 无 | 7/8/22/23/32/33/34 |
| eyes | 28 | 8/23 | 7/22/32/33/34 |
| eyes-neck | 29 | 8/23/32 | 7/22/33/34 |

ID9/24不参加三种播放发送；35轴均参与每遍准备/回中。播放时仍监测33个运行轴，保持不等于不监测。先回中再保持，保持轴不反复发送位置指令。

一次只运行一个示例。控制会话锁覆盖准备与播放；串口使用独占访问。相机进程只发布本机UDP目标，配置不匹配的已有监控服务拒绝复用，避免误用另一只眼的图像。自行启动的视觉子进程无论成功或异常均清理，不终止外部服务。

## 6. 日志及反馈

每轮保存准备参数、反馈恢复、音频结束时间、指令轴集和控制帧记录。60Hz控制按实际音频时钟取源帧，反馈在剩余时隙中以短请求轮询，损坏样本不伪装为健康反馈。当前仍保留通信失败率及反馈失联保护。

软件负载大于800告警，温度大于80℃告警；同轴连续4次有效高温样本大于100℃触发该温度策略停止。设备自身温度/过载保护保持，通信、位置、模式等错误仍可能按既有规则停止。软件阈值不是设备额定温度或允许长期运行范围，现场应结合读回和日志分析。
