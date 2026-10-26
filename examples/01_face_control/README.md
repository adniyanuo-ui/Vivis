# 示例1：面部控制

版本0.2｜2026-10-26｜新入口待现场验收。先阅读[快速上手](../../docs/developer/QUICKSTART.md)。

## 1. 本例做什么

不启动相机；正式播放仅面部表情与音频。 输入为wz2的音频与派生目标轨迹；默认完整播放1遍。原始表情和音频不修改，不按停顿增加口部动作。通过`--work 0001`使用另一份完整歌曲。

## 2. 代码如何工作

[`main.py`](main.py)选择`run_example('face')`，三个示例共用[`runtime/developer.py`](../../src/vivis/runtime/developer.py)管理设备和资源，共用[`runtime/show.py`](../../src/vivis/runtime/show.py)执行准备和播放。入口是完整可运行代码，算法实现包含在`src/vivis/`中，不是外部闭源服务。

1. 读取`Robot`：最终标定、硬件和跟踪配置；从作品清单核对音频/轨迹哈希。
2. `prepare`生成本模式控制权，保持轴在播放中不发送位置指令。
3. 每遍重新检查、恢复参数，35轴缓慢回中，恢复任务速度。
4. 由音频DAC时钟驱动共享播放循环，统一位置发送和反馈。
5. 结束或异常清理本次资源，保留既有颈部扭矩和舌部卸载策略。

播放发送26轴；跟踪拥有轴：无；保持轴：7/8/22/23/32/33/34。ID9/24不参加作品位置发送。准备/回中会涉及全部35轴，不能把播放阶段的保持理解为本次任务完全不动该轴。

## 3. 检查与运行

在仓库根目录执行：

```bash
./examples/01_face_control/run.sh --check
./examples/01_face_control/run.sh --work wz2 --repeat 1
# 现场确认短作品后，可选择完整歌曲：
./examples/01_face_control/run.sh --work 0001 --repeat 1
```

`--check`不打开硬件。运行配置以`configs/robots/mojia/`为准，实际60Hz/1Mbps、口型增益1.0，保留已接受速度及加减速。当前不自动切500kbps。

## 4. 现场验收

确认完整音频可听、口型/眉毛/眼睑等表情正常；回中完成后眼球和颈部不接收播放位置指令。可以在没有相机的条件下运行，不能把没有监控画面判作故障。

保存输出目录并填写[验收表](../../docs/developer/ACCEPTANCE.md)。日志位于`outputs/developer/时间_模式/`，先检查`performance/summary.json`及每轮的`preparation.json`、`playback_log.json`。Ctrl+C停止并清理资源，但不突然卸载颈部；正常结束或重启不等于硬件断电。

## 5. 如何替换上游

使用[命名表情映射API与作品导入](../../docs/developer/DATA_AND_CONTROL_API.md)接入自己的音频、表情或算法输出。两个已有作品只供回放验收，不需要运行LLM、TTS、CUDA或A2F；完整AI交互应用可在相同表情接口之上开发。
