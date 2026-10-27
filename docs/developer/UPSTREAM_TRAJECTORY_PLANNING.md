# 上游轨迹规划（可选，默认开启）

2026-10-27｜用户确认撤销16计数发送死区后分段卡顿消失，P降至16后抖动降低。本功能在此基线上实现，仅作用于示例3的ID32，供现场对照选择，不是运行三个示例的必需组件。

## 1. 安装与开关

当前Jetson已安装并验证依赖。新环境执行`./tools/setup.sh`，现在默认安装轨迹依赖，保证默认开启的示例3可用；原`--trajectory`安装选项保留兼容。依赖固定为`ruckig==0.19.4`，MIT许可，本Jetson的Python3.12/aarch64从源码编译成功。若系统缺少编译工具，先安装`build-essential`和`cmake`。安装不访问机器头。自行选择不安装轨迹依赖的开发者须显式使用`off`。

开启：

```bash
./examples/03_face_eye_neck_tracking/run.sh --work 0001 --repeat 1 --upstream-trajectory on
```

关闭，使用原链路：

```bash
./examples/03_face_eye_neck_tracking/run.sh --work 0001 --repeat 1 --upstream-trajectory off
```

当前默认`neck.upstream_trajectory.enabled=true`，示例3不加选项即开启，也可以显式传`on`。`--upstream-trajectory off`优先于配置并明确关闭。启动横幅、`tracking_effective.json`、`performance/config.json`和`summary.json`记录实际开关。开关在启动时生效，需要更换链路时先Ctrl+C再重新启动，不支持运行中热切换。

示例1/2和`--home-only`不启用，显式传`on`会在打开设备前报错。依赖缺失时说明安装方法或如何关闭，关闭路径不导入Ruckig；规划异常沿用现有停止清理流程，不静默切换算法。

## 2. 两条链路与参数

| 项目 | 关闭 | 开启 |
|---|---|---|
| ID32跟踪参考 | 原控制器积分位置＋150ms位置低通 | 期望速度进入jerk受限在线轨迹生成器 |
| 回中 | 原至少3秒五次曲线＋低通 | 同一规划器生成至少3秒的静止回中轨迹 |
| 状态 | 原流程 | 连续保存浮点位置、速度、加速度 |
| 发送 | 原整数处理、重复目标省略 | 每周期取整、重复目标省略，不按16计数合并 |
| 硬件、眼球、表情 | 当前基线 | 完全相同 |

实现位于[`motion/upstream.py`](../../src/vivis/motion/upstream.py)，接入[`vision/pursuit.py`](../../src/vivis/vision/pursuit.py)和[`motion/ownership.py`](../../src/vivis/motion/ownership.py)。开启时绕过ID32的150ms位置低通及旧量化调度器，不叠加两种平滑。图像端滤波、视觉参与门槛和眼颈控制权保持。

速度上限来自唯一硬件配置600计数/秒。`tracking.json`中初始对照参数为软件加速度800计数/秒²、jerk8000计数/秒³，并非实测最优值；它们与41的原始值128分开。ID32 PID16/32/0、硬件速度600、硬件加速度128、硬件死区0/0保持不变；跟踪范围已同步为用户确认的新EPROM窗口1706–2388。本规划器不改写硬件限位，不恢复原64限制，不引入snap。

跟踪使用现有控制器的期望速度，避免将每帧人脸位置当成一个需要停住的新终点。换向、丢脸回中和重新捕获继承参考p/v/a；回中轨迹在进入该模式时生成一次，后续执行同一轨迹，不每帧重启3秒计时。规划器核验分段位置极值和后续制动轨迹，在位置边界前制动；不以编码器噪声逐帧覆盖参考状态。

浮点参考受速度、加速度、jerk约束；串口目标是整数，有限差分仍可能有量化尖峰。实际舵机受内部位置环和曲线规划、摩擦、齿隙、负载影响，不能声称真实机械jerk也严格满足这些限制。41=128的有效物理加速度尚未完全确认，软硬件约束须继续实测匹配。

## 3. 离线对照证据

冻结用户P16后的完整0001日志`20261027_140437_258584_eyes-neck/performance/round_1/playback_log.json`，6774帧，使用相同的期望速度和回中标志回放4组参数。原相机输入不会随新规划变化，不能预测改变后的真实视觉闭环效果。

| 软件加速度 | jerk | 合成速度阶跃达到50% | 达到90% |
|---:|---:|---:|---:|
| 400 | 4000 | 183ms | 283ms |
| 800 | 8000 | 117ms | 183ms |
| 800 | 16000 | 100ms | 150ms |
| 1600 | 16000 | 83ms | 133ms |
| 原150ms位置低通 | 无显式jerk限制 | 117ms | 350ms |

阶跃为0→100计数/秒、60Hz浮点参考。800/8000达到10%约50ms，原低通约17ms，柔和起步也有响应代价，并非所有阶段更快。先选800/8000用于现场对照，兼顾参考平顺与响应，不能据此认定最优。

该候选在固定输入中位置约2034.92–2437.74，参考速度峰值126.84、加速度峰值800、平均jerk峰值8000，最终回到2047。单步规划耗时P99约0.069ms，最大约0.125ms，是一次离线记录，不是实时性保证。与原低通目标的平均位置差约7.06计数、P95约32.21计数，包含不同回中和状态接续造成的参考差异，不是实测跟踪误差。

本机结果在`outputs/diagnostics/upstream_trajectory_20261027/`，含`comparison.json`和所选方案逐帧`selected_reference.csv`，不作为随包素材。可独立重放自己的日志：

```bash
.venv/bin/python tools/replay_upstream_trajectory.py \
  --log outputs/developer/某次会话/performance/round_1/playback_log.json \
  --output outputs/diagnostics/upstream_comparison
```

开启后每帧`person_gaze.upstream_trajectory`记录参考位置/速度/加速度、平均jerk、周期、请求速度、模式和回中时长；`events.goals`结合`transmitted_ids`核对实际发送，编码器仍来自反馈。

## 4. 验收与回退

依赖、素材、六组默认入口和显式关闭入口检查及197项回归通过。测试覆盖默认开启、显式关闭一致性、依赖缺失时不访问硬件、仅改ID32、浮点参考约束、不规则周期、两侧边界、换向、回中计时、重新捕获和模拟联合播放。完整交付自检需要安装默认依赖；单独运行且跳过原生规划测试的结果不能当作开启路径已验收。

用户随后手动播放0001，会话`20261027_144834_416365_eyes-neck`完整结束112.894秒，6774帧均记录规划器开启。所有帧的`planned_neck_goal`、`smoothed_neck_goal`和`upstream_trajectory.position`完全一致；结合开启分支不创建`NeckCommandScheduler`，确认本次绕过150ms位置低通。保存的`target_filter_tau_s=0.15`只供`off`恢复原链路，数值存在不表示本轮执行；图像误差端的120ms滤波仍保留，是不同处理层。

本次仅改默认开关并保存参数，未访问硬件或自动播放。实际运动品质仍以用户观察为准，不能用完整结束替代抖动验收。重点比较启动、缓慢左右跟随、换向、丢脸回中及重新捕获；若效果不明显或更迟，显式使用`off`恢复原链路即可。

实现依据：[Ruckig官方文档](https://docs.ruckig.com/)、[速度接口与动态跟踪说明](https://docs.ruckig.com/tutorial.html)、[0.19.4源码与许可](https://github.com/pantor/ruckig/tree/v0.19.4)。使用本地单轴状态规划，不调用云端路径服务或Pro Tracking接口。
