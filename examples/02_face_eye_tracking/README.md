# 示例2：面部＋横向眼球跟踪

版本0.2｜2026-10-26｜新入口待现场验收。先阅读[快速上手](../../docs/developer/QUICKSTART.md)。

## 1. 本例做什么

增加人脸检测与横向双眼跟踪，正式播放不发送颈部位置。 输入为wz2的音频与派生目标轨迹；默认完整播放1遍。原始表情和音频不修改，不按停顿增加口部动作。通过`--work 0001`使用另一份完整歌曲。

## 2. 代码如何工作

[`main.py`](main.py)选择`run_example('eyes')`，三个示例共用[`runtime/developer.py`](../../src/vivis/runtime/developer.py)管理设备和资源，共用[`runtime/show.py`](../../src/vivis/runtime/show.py)执行准备和播放。入口是完整可运行代码，算法实现包含在`src/vivis/`中，不是外部闭源服务。

1. 读取`Robot`：最终标定、硬件和跟踪配置；从作品清单核对音频/轨迹哈希。
2. `prepare`生成本模式控制权，保持轴在播放中不发送位置指令。
3. 每遍重新检查、恢复参数，35轴缓慢回中，恢复任务速度。
4. 由音频DAC时钟驱动共享播放循环，统一位置发送和反馈。
5. 结束或异常清理本次资源，保留既有颈部扭矩和舌部卸载策略。

播放发送28轴；跟踪拥有轴：8/23；保持轴：7/22/32/33/34。ID9/24不参加作品位置发送。准备/回中会涉及全部35轴，不能把播放阶段的保持理解为本次任务完全不动该轴。

## 3. 检查与运行

在仓库根目录执行：

```bash
./examples/02_face_eye_tracking/run.sh --check
./examples/02_face_eye_tracking/run.sh --work wz2 --repeat 1
# 现场确认短作品后，可选择完整歌曲：
./examples/02_face_eye_tracking/run.sh --work 0001 --repeat 1
```

`--check`不打开硬件。运行配置以`configs/robots/mojia/`为准，实际60Hz/1Mbps、口型增益1.0，保留已接受速度及加减速。当前不自动切500kbps。

相机默认右眼，镜头随ID23运动，不能沿用固定在头壳上的相机模型。运行时浏览`http://127.0.0.1:8765/`。纵向ID7/22关闭。

```bash
# 明确指定已核对的相机节点：
./examples/02_face_eye_tracking/run.sh --camera right --camera-device /dev/v4l/by-id/实际节点
```

选择左眼用`--camera left --camera-device 实际节点`，相机参考自动切换为ID8。两只相机须分别核对图像方向和设备节点，本包不同时融合两路图像。默认不改变镜像/旋转，相关值在`tracking.json`中配置。

绿框说明检测到人脸，下方模拟眼睛表示图像方向，不等于舵机编码器位置。真机是否正确跟踪以现场和播放日志为准。

## 4. 现场验收

确认面部效果与示例1一致；人脸左右移动时双眼方向正确，纵向眼球和颈部保持；人脸离开画面后横向双眼自然回中。确认停止后下一次示例能重新打开相机。

保存输出目录并填写[验收表](../../docs/developer/ACCEPTANCE.md)。日志位于`outputs/developer/时间_模式/`，先检查`performance/summary.json`及每轮的`preparation.json`、`playback_log.json`。Ctrl+C停止并清理资源，但不突然卸载颈部；正常结束或重启不等于硬件断电。

## 5. 如何替换上游

使用[命名表情映射API与作品导入](../../docs/developer/DATA_AND_CONTROL_API.md)接入自己的音频、表情或算法输出。两个已有作品只供回放验收，不需要运行LLM、TTS、CUDA或A2F；完整AI交互应用可在相同表情接口之上开发。
