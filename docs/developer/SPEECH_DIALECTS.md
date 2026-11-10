# 实时方言切换

更新：2026-11-10。适用于Qwen接管语音的P1交互态。晚间现场验证发现音色能力限制，见第4节；清单可请求不等于当前声音能地道发出全部方言。

## 1. 支持范围与边界

统一实现普通话及官方列出的20种方言／地方口音，无须为每种方言复制程序。唯一清单为[方言配置](../../configs/perception/speech_dialects.json)：

普通话、广东话、重庆话、东北话、甘肃话、贵州话、浙江话、河北话、河南话、湖北话、湖南话、江西话、宁波话、宁夏话、青岛话、陕西话、山西话、山东话、上海话、四川话、云南话。

[官方支持清单](https://help.aliyun.com/zh/model-studio/qwen-audio-realtime-user-guides)同时涵盖中文语音交互能力，不代表已对每种地方发音和当前男声音色完成验收。浙江话等名称沿用官方口径，不等于覆盖省内每种方言。

输入与输出API语言代码仍为`zh`，没有独立的粤语、四川话参数。通过当前会话指令指定目标方言，并按用户确认选择对应男声音色；发音是否地道由现场试听判断。没有新增独立ASR、语言模型或TTS服务；外国话切换不在本次范围内。

## 2. 实现与生效范围

用户说话 → Qwen理解明确切换意图 → `set_speech_dialect({"dialect":"四川话"})` → 本地校验 → 同音色更新会话指令，换音色则重连云端并确认 → 下一次回复采用新偏好。

普通话保持[实时语音配置](../../configs/perception/realtime_voice.json)的默认龙安洋；方言使用[方言配置](../../configs/perception/speech_dialects.json)唯一指定的许南川。服务器只允许在首次会话更新设置音色，因此普通话↔方言需要一次云端连接切换，方言之间仍复用原连接。麦克风、AEC、扬声器租约、机器人控制进程及舵机不重启，不重复回中。不能把音色切换说成零延迟。

- 默认普通话。粤语归一到广东话，沪语归一到上海话，陕西话与山西话分开。语义理解仍由Qwen负责，没有本地转写关键词执行规则。
- 只在交互态准备完成、且有人脸或显式唤醒许可时接受切换。待机先唤醒，节目中先退出到交互态再切换；停止／待机工具原有权限不变。
- 不截断已输出半句的音频。通常从下一次回复起生效；用户正常插话仍走原有双工打断逻辑。
- 本次访客会话中持续生效，跨节目切换、待机和云端空闲重连续用。云端重连续接近期文字上下文，条数为`max(12, 2 × max_history_turns)`，当前20条、每条最多2000字符，另有原有按人记忆；不宣称完整迁移云端音频或无限历史。预录歌曲、剧本音频保持原样。
- 确认访客离开或换人时恢复配置默认普通话；身份暂不确定只清除个人会话，不重置公共方言偏好。程序重启也恢复默认，不把甲的个人背景带给乙。
- 人设与知识库的标准普通话是默认值；当前有效方言指令覆盖其语言风格，身份、知识和隐私边界不变。偶尔听到用户说方言不自动切换。
- 同音色返回`speech_instruction_updated`只说明指令已发送；换音色先返回`pending_voice_reconnect`并抑制旧音色出声，握手确认后记录`speech_voice_ready`再回复。`pronunciation_verified=false`明确表示没有自动证明实际口音。重复调用不重复执行，迟到、未支持名称和附加音色／任意指令字段均拒绝。

## 3. 现场测试

更新代码后重启一次服务，跟随日志观察识别结果：

```bash
sudo systemctl restart vivis-p1.service
journalctl -u vivis-p1.service --since '1 minute ago' -f
```

依次说：

1. “小帅，接下来用四川话跟我聊天。”然后问“你今天能帮我做些什么？”
2. “换成粤语。”继续聊两三轮，检查没有自行回普通话。
3. “换成东北话，讲个简短笑话。”检查男声音色、语速和口型是否正常。
4. “恢复普通话。”确认正常恢复。
5. 选任意方言后播放白骨夫人，再说停止。歌曲应保持原录音，返回交互态后仍沿用本次方言。

反馈时区分：日志是否正确识别原话、是否请求正确方言、实际声音是否地道、下一轮是否保持、停止节目是否仍及时。当前配置可查询`get_robot_status`的`speech_preference`；运行目录`voice/cloud_events.jsonl`的`speech_dialect_changed`和`cloud_tool`记录请求与结果。文字带方言词不等于语音已经说对。

无设备的合成文字探测可检查模型是否选择正确工具，不验证麦克风识别和实际发音：

```bash
outputs/speech/venv/bin/python tools/probe_cloud_control.py --scenario dialect --dialect 四川话
outputs/speech/venv/bin/python tools/probe_cloud_control.py --scenario dialect --dialect 广东话
outputs/speech/venv/bin/python tools/probe_cloud_control.py --scenario dialect --dialect 普通话
```

探测使用已配置的百炼凭证并产生API调用费用，不打开麦克风、相机、音箱或串口，不执行节目和电机工具。

## 4. 现场发现与音色限制

18:44会话确实调用了四川话、上海话、广东话工具；四川话随后因identity_uncertain重建会话回到默认，是偏好被过早清除。现已将身份不确定宽限3秒改为15秒，且单纯身份不确定不再重置公共方言偏好。另一熟人一经确认仍立即隔离个人上下文。

同时，[官方音色列表](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list)将当前`longanyang_v3.1`（龙安洋）列在精品中文音色中，该组明确仅支持普通话。即使偶尔能说几句粤语，也不能据此保证四川、陕西、上海等发音。上一轮仅用工具选择探测作为准备验收，不能证明语音执行效果。

多语种／方言男声`xunanchuan_v3.1`（许南川）官方明确列出上海、广东、东北、重庆、陕西、云南、宁波、甘肃8类；这份音色说明没有承诺四川话或其余全部20类。实时模型实际接受该音色并生成了四川、陕西、上海、广东四种试音WAV，均未自动试听，不将文字或生成成功当作发音验收。用户已确认方言模式使用许南川，普通话保持龙安洋；未明确支持的方言标记`capability=experimental`，包括四川话。

生成试音文件（只调用云端并保存，不打开本地音箱、麦克风或机器头）：

```bash
outputs/speech/venv/bin/python tools/probe_dialect_audio.py --voice xunanchuan_v3.1 --dialect 上海话
outputs/speech/venv/bin/python tools/probe_dialect_audio.py --voice xunanchuan_v3.1 --dialect 陕西话
```

记录中`pronunciation_verified=false`必须等用户试听后才可更新结论。方言换词、音调、男声音色接受度分别反馈；不能只靠提示词承诺不支持的发音能力。
