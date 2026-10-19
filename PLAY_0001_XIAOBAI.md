# 小白 0001 歌曲一键播放

播放指定文件 `outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json`，配套音频为 `0001.wav`，采用最终标定 `xiaobai/xiaobai标定文件.json`。

默认使用已试播的 **60fps、100% 幅度、±5 编码器计数容差、反馈只记录模式**。表情目标仍受原有位置限位及 300 计数/秒限速约束。颈部保持已归位的 2047，仅监测，唱歌帧不驱动颈部动画。反馈只记录不表示硬件保护均已启用；未改写舵机保护参数。

## 直接启动

可在任意目录运行，无需手动激活虚拟环境：

```bash
/home/ubuntu/project/play_0001_xiaobai.sh
```

只检查、30fps 对比或短片段播放：

```bash
/home/ubuntu/project/play_0001_xiaobai.sh --check
/home/ubuntu/project/play_0001_xiaobai.sh --fps 30
/home/ubuntu/project/play_0001_xiaobai.sh --max-seconds 15
```

按 `Ctrl+C` 停止，或在另一终端执行 `touch /home/ubuntu/project/outputs/xiaobai/STOP`。停止发送不会自动卸掉颈部扭矩。脚本保留 STOP 文件，确认可以重启后由操作员移除。

## 启动条件与日志

舵机电源已开启、TTL/RS485 已连接、音箱已连接；颈部已归位并使能。脚本核对 USB 串号、驱动、读写权限、输入时长、标定哈希及颈部保持状态，不会自动重新使能或抬起未准备好的头部。

USB 拔插后若端口/权限改变，运行：

```bash
sudo python3 /home/ubuntu/project/tools/xiaobai_switch_usb_driver.py vendor
```

重新上电导致未使能或脖子下垂时，先托住并完成已有归位/面部启动流程，再播放；不要靠播放脚本强行恢复。

每次运行创建独立的 `outputs/xiaobai/0001_direct_时间戳/`，保存配置、计划、逐帧日志和串口追踪；反馈异常仅记录，原始追踪有容量上限，完整性以追踪内的丢弃事件数为准。稳定配置存于 `xiaobai/playback_profile_0001_observation.json`。
