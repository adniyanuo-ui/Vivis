# 可选本地语音依赖与素材来源

仓库只包含安装脚本、接口和固定来源清单，不提交模型权重、环境或用户录音。`sources.json`记录Hugging Face仓库、提交版本、文件大小及上游SHA256／Git blob摘要；安装时逐个校验，保存上游自带的README和LICENSE。模型许可应按其来源声明核对，不能把运行库许可代替模型许可。

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：可选环境固定2.0.1，流式Zipformer、SenseVoice整句确认和本地Kokoro TTS接口。运行库及其组件许可随Python发行包提供。
- [Zipformer中文流式模型说明](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)：选用2025-06-30的int8版本；权重来源与固定提交见清单。
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：整句候选确认，下载包中的LICENSE引用FunASR许可说明；保留该引用。它不是远程语音服务。
- [Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)和[sherpa转换模型与声音清单](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_1.html)：仅离线生成菜单等短提示，模型包保留Apache-2.0许可。当前使用58号`zm_009`中文男声；代码也保留VITS配置接口，当前安装清单不再下载旧Melo模型。
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)：MIT许可，固定0.1.1，在独立语音环境中测量BS.1770综合响度。系统[SoX](https://sox.sourceforge.net/sox.html)仅在离线准备提示音时按需做动态压缩；不复制其二进制，不调用音箱。当前归一化方案不依赖Jetson精简版FFmpeg的音频滤镜。
- [webrtcvad-wheels](https://github.com/daanzu/py-webrtcvad-wheels)：固定2.0.14，流式20ms语音分段；发行包保留Python封装及WebRTC代码的许可。
- [SpeexDSP](https://www.speex.org/docs/manual/speex-manual/node7.html)：通过系统`libspeexdsp1`动态链接，仓库不复制其二进制或源码。按DAC／ADC时间戳对齐参考并处理采样时钟偏差，效果仍需实际音箱与麦克风双讲验证。
- [pywebrtc-audio](https://github.com/strands-labs/pywebrtc-audio)：纯语音试验环境固定0.2.0，提供WebRTC AEC3及NS的Python接口；AGC关闭。封装声明Apache-2.0，WebRTC及其第三方组件保留各自上游许可。通过pip安装，不将wheel或源码复制进交付仓库；当前ARM64/Python3.12 wheel来源与摘要见`realtime_frontend_sources.json`。此候选只通过显式`--aec-backend aec3`选用，不改P1默认前端。

本地运行不向这些站点上传音频；安装模型时需要下载。现场诊断录音和测试生成的WAV位于已忽略的`outputs/`，不进入Git提交。
