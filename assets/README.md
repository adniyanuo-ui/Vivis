# 随包素材与依赖

本目录内容随Git提供，clone后无需从其他工程恢复文件。

## 1. 两份可直接播放的作品

| 作品 | 音频 | 派生目标 |
|---|---|---|
| 0001 | `assets/works/0001/audio.wav` | `assets/works/0001/targets.json` |
| wz2 | `assets/works/wz2/audio.wav` | `assets/works/wz2/targets.json` |

四个文件总计约9MB。两份音频保留原始哈希；当前目标轨迹已按`墨甲10.29.json`（JawOpen端点按用户确认修正为1547）重新生成，保留原帧数、已接受的眨眼时序及增益，现场效果待复验。两份作品实际只改变ID31目标，其他轴逐值保留。`works.json`记录来源哈希、帧数和位置；电机参数只在`configs/robots/mojia/hardware.json`。素材不是Git LFS指针，普通clone即可取得实际文件。

```bash
.venv/bin/python tools/check_assets.py
```

编辑输入、重新导出或替换素材时同步维护作品清单，不修改哈希来冒充已经验收。作品导入生成到`outputs/imported_works/`；新作品要交给别人时同时纳入其音频/轨迹，不能只交清单。

## 2. 必要第三方组件

- 飞特SMS/STS SDK：`src/vivis/hardware/feetech/_vendor/scservo_sdk/`。保留固定提交的6个实现模块及MIT许可证，精简`__init__.py`为实际使用的导出；无额外SDK下载或全局路径注入。
- YuNet人脸模型：`assets/models/yunet/face_detection_yunet_2023mar.onnx`，约228KB，附MIT许可证与固定版本/哈希清单。示例2/3按哈希核验后加载，不访问网络下载。
- 完整来源和版本：`assets/third_party.json`。Python依赖由`pyproject.toml`安装，不把虚拟环境提交到Git。

预制回放默认不读取原始61通道制作数据。两份作品均随包提供`assets/works/作品名/source.json`，用于显式`--round-lip-gain`运行时通道增益：0001是约8.78MB的原始60FPS数据，wz2是约914KB的原始30FPS数据，原字节与哈希保留；它们不是另一套硬件配置。wz2按原保持采样方式转换到60FPS，0001保持原60FPS；只重新映射Funnel/Pucker涉及的轴，其余已接受轨迹列保留。该功能不依赖原工程目录。

A2F权重不属于预制回放的运行依赖。客户接入自己的素材时可按教程生成新作品。完整AI产品、训练平台和相关模型的分发另行约定。
