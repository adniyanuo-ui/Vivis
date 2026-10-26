# 示例素材交接

Git仅保存`works.json`及来源哈希，不包含大媒体、目标轨迹或第三方模型。仓库源码不是完整素材备份。

## 1. 当前四个必需运行文件

| 作品 | 音频 | 电机目标 |
|---|---|---|
| 0001 | `exmple/0001.wav` | `mojia/0001_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000.json` |
| wz2 | `mojia/wz2_expression15.wav` | `mojia/wz2_expression15_MOJIA_MOTOR_TARGETS_60fps_BLINK_NATURAL1000.json` |

文件名保留历史命名，NATURAL1000不是当前所有轴速度。权威速度在`configs/robots/mojia/hardware.json`，当前14个口型相关轴1200、其余映射轴1000。

交付时另外提供上述4文件的素材包。保留目录结构，或放在同一素材目录中，再执行：

```bash
PYTHONPATH=src .venv/bin/python -m vivis.runtime.assets --from-dir /path/to/supplied-assets
```

已存在文件只核验不覆盖，复制前核验来源SHA256。当前机器可不带`--from-dir`直接检查。

## 2. 可恢复的交付

交接包含源码提交、`configs/`、示例教程、本素材包、固定版本第三方SDK和YuNet安装/许可说明。音频与原始表情素材使用权单独确认；第三方组件按其实际许可分发。

原始61通道数据不是预制回放的必需输入，但客户重新编译作品或复现制作过程时需同时取得匹配原始数据。上游A2F模型和CUDA/TensorRT不是三个回放示例的必要依赖；若交接A2F能力，另行提供其权重、环境及许可证。

舵机SDK按`tools/external-sources.json`恢复，YuNet按`tools/model-manifests/yunet.json`下载并核验。离线环境应预先准备相应目录及Python依赖包。不要将私有合同、账号凭据或整个实验输出目录打入开发者交付包。
