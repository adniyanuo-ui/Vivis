# Qwen-Audio-3.1-Realtime纯语音测试

## 1. 范围

本测试独立运行，只使用DOV麦克风、USB音箱和网络，不打开串口、相机、A2F或节目管理器，不要求机器头上电。现有P1语音切换和运动参数保持不变。

当前先测`qwen-audio-3.1-realtime-plus`，`smart_turn_v2`，正常语速1.0、情绪增强开启。首个音色`longanyang_v3.1`（官方“阳光大男孩”）；候选`longshao_v3.1`、`longze_v3.1`。音色说明不是本站听感验收，需现场试听。人物身份读取选定robot.json，小帅为20岁、北京男青年；提示明确标准普通话，不刻意使用重方言。

## 2. 先配置账号与认证

1. 登录[北京地域API Key管理](https://bailian.console.aliyun.com/cn-beijing/model/settings/api-key)，确认右上角华北2（北京）。
2. 创建通用API Key，归属账号选择自己的主账号，归属业务空间选择默认业务空间；按需要配置模型访问权限。
3. 立即复制创建成功弹窗中的完整密钥。新版Key关闭弹窗后列表只显示打码值，不能用打码值或Key ID代替密钥；参见[官方说明](https://help.aliyun.com/zh/model-studio/get-api-key/)。
4. 不发送密钥给协作者，不放入Git或命令行参数。当前程序可交互读取本次密钥，不保存；也可使用当前终端的`DASHSCOPE_API_KEY`环境变量。

首次准备本地依赖（当前Orin已准备）：

```bash
cd /home/james/project/Vivis-delivery
./tools/setup_realtime_voice.sh
```

AEC复用原有SpeexDSP，系统需已有`libspeexdsp1`。安装脚本不打开音频或连接云端。

先探测，暂不采音／播放：

```bash
./tools/test_realtime_voice.sh --probe --region beijing --ask-key
```

提示时粘贴完整Key并回车，终端不回显。若当前终端已正确export密钥，可省略`--ask-key`。探测成功应显示“连接、模型权限、音色及语义轮次配置已确认”。服务端必须确认请求的voice和轮次类型；不自动退回女声或声学VAD。

如果创建Key时确实选了新加坡，使用`--region singapore`；不自动把密钥发往多个地域猜测。若需要业务空间专属域名，添加`--workspace-id 实际ID`；该ID不是API Key。

遇到401：先看“认证诊断”中的code/message/request_id；`InvalidApiKey`表示当前端点未通过认证，尚未开始模型参数检查。核对密钥完整值、地域、业务空间与服务地址是否匹配，不通过换音色或修改AEC绕过。套餐Key需使用对应套餐的接入方式，不能与通用地址混用。[官方错误码](https://help.aliyun.com/zh/model-studio/error-code)

以下命令只显示公开配置及密钥格式布尔值／长度，不显示密钥，不连接网络或打开设备：

```bash
./tools/test_realtime_voice.sh --check
```

## 3. 开始聊天

先Ctrl+C退出旧P1、歌曲及其他占用麦克风／音箱的程序。保持日常使用的音箱音量及位置，首次站在距麦克风约0.5～1米处，先不要改变摆位。这个距离是测试起点，不是模型保证范围。

```bash
./tools/test_realtime_voice.sh --region beijing --ask-key --seconds 600
```

等待“纯语音聊天已就绪”。直接说话，不需要先喊唤醒词。默认10分钟自动结束；运行会调用百炼在线服务，实际用量记录在日志，未设置任何自动充值或无限重连。

- Enter：本地停声并尝试取消进行中的响应。
- Ctrl+C或输入`/quit`：退出测试，释放音频。
- 输入`/mark echo_only`等：标记测试阶段，只写日志，不修改麦克风或模型行为。

按下面顺序观察，建议总计5～10分钟：

| 阶段 | 你的操作 | 重点反馈 |
|---|---|---|
| 男声与人设 | “介绍一下自己”“讲个短笑话” | 男声是否年轻、普通话是否自然、语速是否正常、是否油腻／机械 |
| 情绪回应 | “今天有点累，事情都不顺” | 是否理解和安慰，有无僵硬追问或强行搞笑 |
| 仅自播 | 先标记`/mark echo_only`，请它讲30秒左右故事，之后你保持安静 | 是否把自己的声音转写成用户话语、自我接话或自行打断 |
| 有效打断 | 标记`/mark barge_in`，它说话中间正常说“等一下，换个话题”或“先别说了” | 多久停声、是否残留旧话、能否接上新话题 |
| 附和与旁人 | 说“嗯”“对”，另选短时段让旁人讲话 | 是否不必要地抢话／停声；记录当时具体说了什么 |

若出现反复自我接话，先Ctrl+C终止该轮并保留日志，不让循环持续消耗服务。

换音色需要新建会话（voice只在首次配置时生效）：

```bash
./tools/test_realtime_voice.sh --ask-key --voice longshao_v3.1 --seconds 180
./tools/test_realtime_voice.sh --ask-key --voice longze_v3.1 --seconds 180
```

按顺序分别运行，不同时运行。当前认可的Qwen3-TTS Dylan缓存没有替换；新云端音色单独验收。

## 4. 回声对照和记录

默认`--aec on`复用现有AEC参数，播放期间持续采音，不用关麦规避双讲。参考来自实际输出回调的PCM，包含音量与退出淡出；按DAC／ADC时间戳对齐后再处理。参数来源仍为speech.json，不修改P1源文件。

如果默认测试可用，再在相同摆位、音量和提示词下做短时原始输入对照：

```bash
./tools/test_realtime_voice.sh --ask-key --aec off --seconds 120
```

关闭AEC时上传下采样的原始麦克风音频，用来观察模型／轮次策略自身的回声表现。对比结果依赖现场条件，不能把“偶尔没自我接话”当作已经消除回声。

默认只保存事件、转写、用量及数值统计，不录制PCM。若愿意保存本次声音供分析，在运行命令后加`--record-audio`，会保存三份16kHz、同帧对齐的WAV：麦克风、实际播放参考、上行音频。只记录本客户端完成发送的帧；这不代表服务端已逐帧确认接收。文件只保存在本机outputs。

## 5. 日志及测量边界

每次输出到`outputs/realtime_voice/时间戳/`：

- `effective.json`：公开有效配置、输入／输出设备选择，不含密钥。
- `events.jsonl`：服务端事件、用户／机器人转写、阶段标记、音频数值、首包和实际DAC时间；不保存Base64音频或请求Authorization。
- `summary.json`：错误、响应记录、用量、回调告警、取消操作／响应数量、拒收旧音频包等；取消操作数不等于用户打断次数。

从服务端`speech_stopped`到首包／首个DAC时间是客户端可观测阶段耗时，不能直接称为用户说完到听见声音的完整端到端延迟。`speech_started`到本地取消／缓冲排空同理，不含上行和服务端检测之前的时间。40ms是客户端淡出配置，不是实测总打断延迟。

当前按官方示例在`speech_started`时清除旧播放队列并淡出，拒绝该response_id的迟到包；语义轮次也可能出现`turn_invalid`。因此现场“嗯”导致停声时，要同时看服务端是否真正cancelled，区分模型轮次判断与客户端即时停声策略。服务端自动打断时不再额外发送response.cancel，避免取消新响应；Enter显式取消单独记录。推理完成事件不会丢弃仍待实际播放的音频。

2026-11-08本地静音检查：5.01秒，约250帧处理，无回调告警，无音频保存／上传；仅证明本地I/O处理链路可运行。初次云端探测返回401 InvalidApiKey；随后20261108_144319_668700探测成功，服务端确认qwen-audio-3.1-realtime-plus、longanyang_v3.1、smart_turn_v2，未打开麦克风或音箱。认证及会话配置已通过，男声听感与真实双讲仍待现场验证。

2026-11-08启动时序修复：`20261108_144600_854053`中设备准备约1.05秒，旧看门狗却从准备前计时，打开后不到0.1秒就误判无回调；三份WAV只有文件头，不能用于听感或回声结论。现将准备与运行时钟分开，从设备打开后监测首次回调，运行中仍按最近回调保持原1秒断流保护。“已就绪”需输入／输出回调及首帧发送均成立。新增准备耗时、实际运行耗时和发送帧数记录。

回归在旧实现复现同一watchdog错误，修复后通过；另以真实DOV／USB音箱、模拟云连接做两次静音运行，每次故意增加1.2秒准备，实际准备约2.277／1.248秒、各发送99帧，均无回调告警。没有云请求、录音或机器头动作；真实云端聊天仍需重试验收。

官方参考：[实时语音指南](https://help.aliyun.com/zh/model-studio/qwen-audio-realtime-user-guides)、[客户端事件](https://help.aliyun.com/zh/model-studio/fun-audiochat-client-events)、[服务端事件](https://help.aliyun.com/zh/model-studio/qwen-audio-realtime-server-events)、[音色表](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list)。
