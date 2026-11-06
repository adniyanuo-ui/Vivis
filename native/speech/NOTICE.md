# 可选本地语音依赖与素材来源

仓库只包含安装脚本、接口和固定来源清单，不提交模型权重、环境或用户录音。`sources.json`记录Hugging Face仓库、提交版本、文件大小及上游SHA256／Git blob摘要；安装时逐个校验，保存上游自带的README和LICENSE。模型许可应按其来源声明核对，不能把运行库许可代替模型许可。

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)：可选环境固定2.0.1，流式Zipformer、SenseVoice整句确认和本地MeloTTS接口。运行库及其组件许可随Python发行包提供。
- [Zipformer中文流式模型说明](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)：选用2025-06-30的int8版本；权重来源与固定提交见清单。
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：整句候选确认，下载包中的LICENSE引用FunASR许可说明；保留该引用。它不是远程语音服务。
- [MeloTTS](https://github.com/myshell-ai/MeloTTS)和[转换模型说明](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)：仅用于本地生成菜单等短提示，下载包保留其MIT许可。
- [webrtcvad-wheels](https://github.com/daanzu/py-webrtcvad-wheels)：固定2.0.14，流式20ms语音分段；发行包保留Python封装及WebRTC代码的许可。
- [SpeexDSP](https://www.speex.org/docs/manual/speex-manual/node7.html)：通过系统`libspeexdsp1`动态链接，仓库不复制其二进制或源码。按DAC／ADC时间戳对齐参考并处理采样时钟偏差，效果仍需实际音箱与麦克风双讲验证。

本地运行不向这些站点上传音频；安装模型时需要下载。现场诊断录音和测试生成的WAV位于已忽略的`outputs/`，不进入Git提交。
