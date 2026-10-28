# 示例3：面部＋眼球＋颈部跟踪

版本0.2｜2026-10-26｜wz2一遍整体运行成功；新ID32转向抖动待处理。先阅读[快速上手](../../docs/developer/QUICKSTART.md)。

## 1. 本例做什么

增加颈部旋转ID32，沿用已接受的眼颈协调算法。可用`--parallel-neck on`增加ID33/34共同实现的俯仰与轻微侧倾，见[并联机构与首轮验证](../../docs/developer/PARALLEL_NECK_TRACKING.md)。输入为wz2的音频与派生目标轨迹，默认完整播放1遍。原始表情和音频不修改，不按停顿增加口部动作。通过`--work 0001`使用另一份完整歌曲。

## 2. 代码如何工作

[`main.py`](main.py)选择`run_example('eyes-neck')`，三个示例共用[`runtime/developer.py`](../../src/vivis/runtime/developer.py)管理设备和资源，共用[`runtime/show.py`](../../src/vivis/runtime/show.py)执行准备和播放。入口是完整可运行代码，算法实现包含在`src/vivis/`中，不是外部闭源服务。

1. 读取`Robot`：最终标定、硬件和跟踪配置；从作品清单核对音频/轨迹哈希。
2. `prepare`生成本模式控制权，保持轴在播放中不发送位置指令。
3. 每遍重新检查、恢复参数，35轴缓慢回中，恢复任务速度。
4. 由音频DAC时钟驱动共享播放循环，统一位置发送和反馈。
5. 结束或异常清理本次资源，保留既有颈部扭矩和舌部卸载策略。

播放发送29轴；跟踪拥有轴：8/23/32；保持轴：7/22/33/34。ID9/24不参加作品位置发送。准备/回中会涉及全部35轴，不能把播放阶段的保持理解为本次任务完全不动该轴。

开启并联平台后播放发送31轴，跟踪独占8/23/32/33/34，7/22继续保持。视频仍为右眼，32及眼球的原运动参数不变；33短拉杆、34长拉杆按CAD闭链共同求解，不能当成两个独立姿态轴。

## 3. 检查与运行

在仓库根目录执行：

```bash
./examples/03_face_eye_neck_tracking/run.sh --check
./examples/03_face_eye_neck_tracking/run.sh --work wz2 --repeat 1
# 现场确认短作品后，可选择完整歌曲：
./examples/03_face_eye_neck_tracking/run.sh --work 0001 --repeat 1
```

`--check`不打开硬件。运行配置以`configs/robots/mojia/`为准，实际60Hz/1Mbps、口型增益1.0，保留已接受速度及加减速。当前不自动切500kbps。

相机默认右眼，镜头随ID23运动，不能沿用固定在头壳上的相机模型。运行时浏览`http://127.0.0.1:8765/`。纵向ID7/22关闭。

```bash
# 明确指定已核对的相机节点：
./examples/03_face_eye_neck_tracking/run.sh --camera right --camera-device /dev/v4l/by-id/实际节点
```

选择左眼用`--camera left --camera-device 实际节点`，相机参考自动切换为ID8。两只相机须分别核对图像方向和设备节点，本包不同时融合两路图像。默认不改变镜像/旋转，相关值在`tracking.json`中配置。

绿框说明检测到人脸，下方模拟眼睛表示图像方向，不等于舵机编码器位置。真机是否正确跟踪以现场和播放日志为准。

眼球先响应图像误差。偏差继续增长且眼球已有同向运动反馈时，颈部参与；对随眼球转动的相机，还允许“眼球实测已使用25%行程、正在同向运动且图像仍有明显偏差”触发颈部辅助，不必等到眼球饱和。两侧使用同一条件，反馈过期时不据此招募颈部；反向时清除旧方向图像误差残留，没有固定等待延迟。

软件64加减速限制已移除，运动加减速由SRAM41控制，当前32号为128，最大运行速度46为600。PID当前16/32/0（仅P从32降为16），硬件死区0/0及跟踪范围1600–2600保持。

颈部目标低通时间常数为`neck.target_filter_tau_s=0.15`秒，收到新目标立即响应并渐近跟随，不是固定等待150ms。16计数发送死区因现场出现明显分段卡顿已撤销，恢复逐周期计算变化目标，不再累计16计数后发送。原一计数量化滞回及整数目标不变时省略重复发送的行为保持。

丢脸或歌曲结束回中采用起止速度均为0的平滑曲线，`neck.return_duration_s=3.0`秒是参考轨迹最短时长；低速度上限会延长时间，低通还需要收尾。轨迹从当前已滤波的命令位置开始，不宣称该值是编码器实测位置。眼球参数不变。

用户确认撤销16发送死区后卡顿消失，P16后抖动降低。当前可选的[上游轨迹规划](../../docs/developer/UPSTREAM_TRAJECTORY_PLANNING.md)已改为默认开启；安装脚本默认包含依赖，可做对照：

```bash
# 新环境首次需要：./tools/setup.sh
./examples/03_face_eye_neck_tracking/run.sh --work 0001 --repeat 1 --upstream-trajectory on
# 先Ctrl+C结束当前轮，再关闭该功能启动另一轮
./examples/03_face_eye_neck_tracking/run.sh --work 0001 --repeat 1 --upstream-trajectory off
```

开启仅接管ID32目标平滑，绕过150ms位置低通，生成速度/加速度/jerk受限的浮点参考；软件加速度800计数/秒²、jerk8000计数/秒³，与硬件128的含义不同。关闭恢复上文原链路；不加选项现在默认开启。PID、硬件参数、眼球及作品不变。开关在启动时生效，若效果不明显或更迟就用`off`。已核验本次0001完整播放6774帧规划器均开启，未叠加150ms低通；该字段只为`off`回退保留。现场现象见[颈部诊断记录](../../docs/developer/NECK_JITTER_DIAGNOSIS.md)。

## 4. 现场验收

确认面部效果保持，眼球与颈部共同追踪人脸，方向正确、转向平顺，无明显抖动；人脸离开画面后眼颈自然回中。与示例2比较，新增运动应仅来自ID32，不应改变口型或新增纵向眼球运动。

保存输出目录并填写[验收表](../../docs/developer/ACCEPTANCE.md)。日志位于`outputs/developer/时间_模式/`，先检查`performance/summary.json`及每轮的`preparation.json`、`playback_log.json`。Ctrl+C停止并清理资源，但不突然卸载颈部；正常结束或重启不等于硬件断电。

## 5. 如何替换上游

使用[命名表情映射API与作品导入](../../docs/developer/DATA_AND_CONTROL_API.md)接入自己的音频、表情或算法输出。两个已有作品只供回放验收，不需要运行LLM、TTS、CUDA或A2F；完整AI交互应用可在相同表情接口之上开发。

## 圆唇表现对照

若圆唇、嘟唇不明显，可在保持硬件参数和原素材时测试`--mouth-command-mode lips-direct`。它只改变10个圆唇相关轴在音乐阶段的目标发送方式，默认仍为`limited`；见[分析与对照说明](../../docs/developer/LIP_FULLNESS_COMPARISON.md)。
