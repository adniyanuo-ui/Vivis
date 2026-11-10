# 小帅系统提示音：晨煦

2026-11-08由用户使用百炼账户生成。音色为`Ethan`（晨煦），模型`qwen3-tts-flash`，中文、原生语速1.0。用于P1本地固定播报；播放不加载TTS模型，也不请求云端合成。

- `welcome.wav`：唤醒应答。
- `menu.wav`：能力及节目介绍。
- `clarify.wav`：节目选择澄清。
- `standby.wav`：进入待机。
- `interaction.wav`：返回交互态。
- `identity.wav`：身份介绍。
- `fault.wav`：故障提示。

全部成品为24kHz、单声道PCM16标准WAV。`texts.json`保存文本，`metadata.json`保存音色、来源清单哈希、每条音频SHA256、实际时长对应帧数及响度处理记录。目标响度相对作品0001低3LU，真峰值不超过-1dBFS；短提示受峰值限制时，实际响度见记录。

云端源文件采用未知大长度的流式WAV文件头，成品重新封装为实际长度；不需要下载2GB文件。原始下载位于本机忽略目录`outputs/speech/cloud_prompt_sources/`，交付播放只需要本目录成品及对应配置。

来源：[官方音色表](https://www.alibabacloud.com/help/tc/model-studio/qwen-tts-voice-list)、[官方合成API](https://help.aliyun.com/en/model-studio/qwen-tts-api)。重生成及启用步骤见[实时语音说明](../../../docs/developer/P1_REALTIME_VOICE.md)。
