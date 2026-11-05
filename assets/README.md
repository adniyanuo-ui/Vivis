# 随包素材与依赖

本目录内容随Git提供，clone后无需从其他工程恢复文件。

## 1. 四份可直接播放的作品

| 作品 | 音频 | 派生目标 |
|---|---|---|
| 0001 | `assets/works/0001/audio.wav` | `assets/works/0001/targets.json` |
| wz2 | `assets/works/wz2/audio.wav` | `assets/works/wz2/targets.json` |
| Face_ControlBoard_CtrlRig（作品3） | `assets/works/Face_ControlBoard_CtrlRig/audio.wav` | `assets/works/Face_ControlBoard_CtrlRig/targets.json` |
| demo3（作品4） | `assets/works/demo3/audio.wav` | `assets/works/demo3/targets.json` |

0001、wz2的音频及派生目标总计约9MB。两份音频保留原始哈希；目标轨迹已按`墨甲10.29.json`（JawOpen端点按用户确认修正为1547）重新生成，保留原帧数、已接受的眨眼时序及增益；该轮只改变ID31目标，其他轴逐值保留。`works.json`记录来源哈希、帧数和位置；电机参数只在`configs/robots/mojia/hardware.json`。素材不是Git LFS指针，普通clone即可取得实际文件。

作品3来自`incoming/demo2/`的61通道JSON和PCM16 WAV：2010帧、30FPS表情数据按逐帧保持生成4020帧、60FPS轨迹，播放共67秒。原JSON逐字节保留为`source.json`；原音频长66.005333秒，播放副本仅在末尾补0.994667秒静音，原音频内容和表情时序保留。处理记录与原始哈希见`assets/works/Face_ControlBoard_CtrlRig/provenance.json`。作品3已登记源文件及哈希，可使用`--round-lip-gain 1.2`；用户反馈上电后真机播放成功。

作品4用`--work demo3`选择，来自`incoming/demo3/9.11（16.50）.json`和同名WAV。原JSON标注30FPS，但848帧只有28.266667秒，音频长35.387211秒。用户授权按表情保真优先选择时间基准后，采用推定的`24000/1001`（约23.976）FPS：848帧对应35.368667秒，与音频仅差约18.5毫秒；该帧率未获原作者确认，不将其描述为已测得的真实帧率。

作品4将全部848个原始表情帧按逐帧保持重采样为2124帧、60FPS，轨迹覆盖35.4秒，完整音频原字节保留。每个原始帧均被使用，重采样不更改权重或峰值；尾部保持最后一帧以覆盖音频。原始JSON逐字节保存在`assets/works/demo3/original_source.json`，运行源`source.json`移除空的`audioFile`元数据并保存重采样数据。处理依据、哈希和复现公式见同目录`provenance.json`。支持`--round-lip-gain 1.2`，控制代码及设备参数未变，现场表现待确认。

```bash
.venv/bin/python tools/check_assets.py
```

编辑输入、重新导出或替换素材时同步维护作品清单，不修改哈希来冒充已经验收。作品导入生成到`outputs/imported_works/`；新作品要交给别人时同时纳入其音频/轨迹，不能只交清单。

## 2. 必要第三方组件

- 飞特SMS/STS SDK：`src/vivis/hardware/feetech/_vendor/scservo_sdk/`。保留固定提交的6个实现模块及MIT许可证，精简`__init__.py`为实际使用的导出；无额外SDK下载或全局路径注入。
- YuNet人脸模型：`assets/models/yunet/face_detection_yunet_2023mar.onnx`，约228KB，附MIT许可证与固定版本/哈希清单。示例2/3按哈希核验后加载，不访问网络下载。
- 完整来源和版本：`assets/third_party.json`。Python依赖由`pyproject.toml`安装，不把虚拟环境提交到Git。

预制回放默认不读取61通道制作数据。四份作品均随包提供`assets/works/作品名/source.json`，用于显式`--round-lip-gain`运行时通道增益：0001是约8.78MB的原始60FPS数据，wz2是约914KB的原始30FPS数据，作品3是约1.32MB的原始30FPS数据，这三份源文件的原字节与哈希保留；作品4使用上述派生60FPS数据，并另存原始JSON。它们不是另一套硬件配置。30FPS源数据按逐帧保持转换到60FPS，0001保持原60FPS；只重新映射Funnel/Pucker涉及的轴，其余轨迹列保留。该功能不依赖原工程目录。

A2F权重不属于预制回放的运行依赖。客户接入自己的素材时可按教程生成新作品。完整AI产品、训练平台和相关模型的分发另行约定。

## 3. 机器人专属轨迹

上表四份作品属于默认墨甲清单`assets/works.json`。小帅有独立清单`configs/robots/xiaoshuai/works.json`，作品1复用`assets/works/0001/`的原音频与源JSON，按小帅标定和设备范围重新计算的目标位于`assets/works/xiaoshuai/0001/targets.json`。运行时通过`--robot configs/robots/xiaoshuai/robot.json`选择，两个机器人的电机轨迹和硬件参数不混用。

```bash
.venv/bin/python tools/check_assets.py --robot configs/robots/xiaoshuai/robot.json
```
