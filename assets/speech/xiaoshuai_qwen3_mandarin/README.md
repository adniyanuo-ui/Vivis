# 小帅普通话系统播报缓存

七条PCM16／24kHz音频由Qwen3-TTS-12Hz-0.6B-CustomVoice生成，使用Dylan音色、Chinese语言及`force_standard_chinese=true`。模型默认给Dylan附加北京方言标签；本次在生成期间仅修改内存中的语言标签选择，保留其音色，随后恢复原配置。下载的权重和配置文件未改写。

用户2026-11-08现场试听后接受普通话标签版。原生语速1.0，无加速或降调；模型按完整提示句生成，保留自身标点韵律，不叠加旧Kokoro的固定分句停顿。响度沿用`configs/perception/speech.json`中的相对作品音频−3LU和真峰值限制。

`metadata.json`记录提示文本、模型与来源清单哈希、响度处理、每条WAV哈希及实际生成环境。整套缓存由RTX4090／FP32生成；Orin本地CPU／FP32已验证同一模型可生成试听。两种计算设备不承诺逐样本一致。静态音频可移植，**客户播放时不需要4090、Torch或2.5GB的TTS模型**。

```bash
.venv/bin/python -m vivis.speech.prompts --check-cache
```

该检查不打开模型或硬件。语音识别和人脸检测的可选依赖仍按P1教程安装。只有修改提示文本、音色或生成参数，才需要`tools/setup_qwen_tts.py`准备独立环境并重新生成；详情见[语音更新与USB故障复盘](../../../docs/developer/P1_VOICE_AND_USB_20261108.md)。

来源：[Qwen官方代码](https://github.com/QwenLM/Qwen3-TTS)、[ModelScope官方模型](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)。模型项目声明Apache-2.0，文件来源及校验值保存在`native/speech/qwen3_tts_sources.json`。本目录不包含模型权重或用户麦克风录音。
