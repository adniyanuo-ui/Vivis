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
| `telemetry.json` | 软件负载/温度告警与按轴、按故障类型连续确认的播放保护策略 |
| `assets/works.json` | 音频及派生轨迹的路径和来源哈希，无硬件配置副本 |

`Robot`按ID生成运行profile、35轴回中清单、启动寄存器设置和跟踪观察参数；运行日志保存其有效值。作品和运行参数分离，不保留另一套作品profile或历史配置。

设备寄存器范围与任务运动范围分别保存。用户2026-10-27确认ID32新增EPROM限位1706–2388，连续三次读回一致；`device_min/max`和同轴`tracking_min/max`均同步为1706–2388。`tracking.json`选择`range_mode="configured"`后从硬件配置取这两个端点，不把设备全编码器范围当作实际机构行程。启动检查仍要求设备限位与保存值一致，不自动改写限位寄存器。

改变速度以`hardware.json`为准；实际启动会读回核验。改变标定需重新生成受影响轨迹并核对其来源哈希，不能仅修改哈希使旧轨迹通过校验。波特率修改必须主机和35轴一致，当前示例不会自动写设备波特率。

关闭上游轨迹规划时，颈部目标平滑时间常数`neck.target_filter_tau_s`为0.15秒；回中参考最短时长`neck.return_duration_s`为3秒。它们在`tracking.json`定义，不恢复原64计数/秒²软件加速度限制。`person_gaze`日志分别保存`planned_neck_goal`、`smoothed_neck_goal`和`neck_return_duration_s`；关闭规划时前两者表示低通前后位置。`events.goals`是交给发送器的整数请求，须结合`transmitted_ids`确认实际发送的轴。这些均不等于编码器位置。当前32号SRAM41=128仍按唯一硬件配置恢复。

16计数发送死区已因现场分段运动撤销，相关配置、SDK选项及运行日志字段已移除；旧日志中的`goal_deadband`仅属于当时的试验版本。ID32 PID从唯一硬件配置取得16/32/0。

[上游轨迹规划](UPSTREAM_TRAJECTORY_PLANNING.md)在示例3默认开启，配置`neck.upstream_trajectory.enabled=true`；`--upstream-trajectory off`显式关闭，`on`可显式开启。开启时用单轴jerk受限规划接管ID32的位置低通，`planned_neck_goal`/`smoothed_neck_goal`均为规划输出，不表示再次滤波；原始期望速度和参考p/v/a、平均jerk、模式、周期在`person_gaze.upstream_trajectory`记录。150ms字段保留供关闭时回退，开启时不生效；关闭路径不导入Ruckig。开关只在启动时解析，不支持运行中热切换。

`neck.eye_assist_ratio=0.25`定义当前相机眼的实测行程使用率辅助门槛，不是固定等待时间。`person_gaze.eye_offset_command`与`eye_offset_measured`分别记录命令和实测归一化眼位，`eye_feedback_age_s`记录反馈年龄；`eye_reserve`表示图像仍有偏差且已消耗眼部行程，`eye_feedback_stale`表示反馈过期而不请求颈部辅助。没有新增串口轮询，复用现有眼球反馈。

`--mouth-gain`沿用既有6轴偏移倍率，适用ID12/13/27/28/29/31，在`hardware.json`的`runtime.mouth_gain_ids`中列明；它与14个口型相关轴的速度分组是两个概念。默认倍率1.0，不改变目标幅度。

## 5. 控制权与资源

| 模式 | 播放发送轴数 | 跟踪拥有轴 | 播放保持 |
|---|---:|---|---|
| face | 26 | 无 | 7/8/22/23/32/33/34 |
| eyes | 28 | 8/23 | 7/22/32/33/34 |
| eyes-neck | 29 | 8/23/32 | 7/22/33/34 |
| eyes-neck＋parallel-neck on | 31 | 8/23/32/33/34 | 7/22 |

ID9/24不参加三种播放发送；35轴均参与每遍准备/回中。播放时仍监测33个运行轴，保持不等于不监测。先回中再保持，保持轴不反复发送位置指令。

`--parallel-neck on`仅为示例3新增俯仰/轻微侧倾，默认关闭；独立配置及CAD闭链见[并联颈部跟踪](PARALLEL_NECK_TRACKING.md)。`person_gaze.parallel_neck`记录新的双轴参考、测量姿态估计和反馈年龄。YuNet目标的可选`eye_line_roll_rad`是图像双眼连线角，缺失/不可靠时不影响原x/y跟踪。33/34当前速度1000、硬件加速度0保持，新双轴参考另限速100、加速度400和jerk4000。

一次只运行一个示例。控制会话锁覆盖准备与播放；串口使用独占访问。相机进程只发布本机UDP目标，配置不匹配的已有监控服务拒绝复用，避免误用另一只眼的图像。自行启动的视觉子进程无论成功或异常均清理，不终止外部服务。

## 6. 日志及反馈

每轮保存准备参数、反馈恢复、音频结束时间、指令轴集和控制帧记录。60Hz控制按实际音频时钟取源帧，反馈在剩余时隙中以短请求轮询，损坏样本不伪装为健康反馈。当前仍保留通信失败率及反馈失联保护。

软件负载大于800、温度大于80℃仅告警。用户确认硬件限位及扭矩保护已由标定同事验证后，歌曲播放阶段采用以下二级软件保护：

| 故障类型 | 告警条件 | 软件停播条件 |
|---|---|---|
| 电压 | 应答/状态寄存器BIT0，或电压读数超出该轴配置范围 | 同轴连续4个有效完整样本 |
| 电流 | 应答/状态寄存器BIT3 | 同轴连续4个有效完整样本；不新增电流数值阈值 |
| 负载过载 | 应答/状态寄存器BIT5 | 同轴连续4个有效完整样本；软件负载>800仍仅告警 |
| 温度 | 应答/状态寄存器BIT2，或温度读数>100℃ | 同轴连续4个有效完整样本；软件温度>80℃仍仅告警 |

状态位定义已核对[飞特STS内存表](https://ftservo.github.io/ftservo-wiki/reference/parameter/memory-sts/)，与随包SDK定义一致；磁编码器及未定义位不纳入这次延迟确认。

同一轴不同故障分开计数；该类故障恢复正常即清零，其他轴的报警不累加。坏包、超时、未完成样本不增加也不清零计数，仍由原通信保护处理。一个完整样本包含多次短读取，这些应答中的状态位会合并判断，但只算一次观察；原状态寄存器与`packet_status_bits`分别保留，不把报警数据伪造成正常。

在既有帧间预算中优先复查待确认轴，`sensor_fault_confirmation_period_s=0.02`是优先调度间隔目标，实际取决于总线时隙，不是固定停播时延。达到通信失联紧迫期的其他轴优先，读取请求数预算不变；确认过程不额外休眠、阻塞命令循环。完整有效通信更新链路新鲜度，传感器故障由独立计数器判断，两个概念分开。

配置来自唯一`configs/robots/mojia/telemetry.json`。`sensor_fault_confirmation`显式开启此策略；电压、电流、过载确认次数允许2～5，当前均为4；温度次数沿用4。关闭此开关会回到原播放故障策略。启动快照、35轴回中、收舌恢复与首帧接近不启用这项放宽，仍按原检查；磁编码器/未知状态位、位置越界、非法目标、通信失败率/失联、控制循环超时和音频故障也保留原处理。

`playback_log.json.sensor_fault_confirmations`记录每次告警、连续次数、原始反馈与恢复状态；`sensor_fault_policy`记录有效策略。单次异常后任务可继续，但不能据此认定供电或舵机本身已正常。舵机自身保护始终独立生效，软件不修改保护寄存器；退出任务仍采用原保持目标/颈部扭矩与卸载舌部的清理方式。
