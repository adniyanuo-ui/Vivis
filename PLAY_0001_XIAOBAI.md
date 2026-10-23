# 小白 0001 歌曲一键播放

播放指定文件 `outputs/0001_CLAIRE_SINGING_GAZE_v3.0-Claire_BLENDSHAPES_61_60fps.json`，配套音频为 `0001.wav`，采用最终标定 `xiaobai/xiaobai标定文件.json`。

默认使用已试播的 **60fps、100% 幅度、±5 编码器计数容差、反馈只记录模式**。表情目标仍受原有位置限位及 300 计数/秒限速约束。颈部保持已归位的 2047，仅监测，唱歌帧不驱动颈部动画。反馈只记录不表示硬件保护均已启用；未改写舵机保护参数。

## 直接启动

### 已认可的眼球与转头跟随效果

固定配置入口 `play_0001_xiaobai_tracking.sh` 保存 2026-10-19 现场认可效果：60fps、100%面部幅度，眼球 8/23 跟随，颈轴32中位±160计数，三倍转头速度（75计数/秒），软件加减速70计数/秒²。先启动视觉服务，在另一个终端播放：

```bash
./track_person_xiaobai.sh
# 另一个终端；视觉服务已运行时无需重复启动
./play_0001_xiaobai_tracking.sh --check
./play_0001_xiaobai_tracking.sh
```

预览地址 `http://127.0.0.1:8765/`。摄像头更换USB口后，停止旧视觉服务（其终端按Ctrl+C）并重新运行，再刷新网页；稳定by-id路径保持不变时无需改配置。原进程不会自动重连。

需连续播放三遍时：

```bash
for round in 1 2 3; do
  ./play_0001_xiaobai_tracking.sh || break
  if [ "$round" -lt 3 ]; then sleep 5; fi
done
```

### 原歌曲编排及参数对比

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

新增人脸水平跟随开关 `--track-person`，启用前先运行 `track_person_xiaobai.sh` 并完成摄像头及电机方向确认。安装、预览和分步验收见[人脸跟随方案](docs/PERSON_GAZE_V1.0.md)；不开此开关保持原眼神编排。

整首歌曲中组合眼球与脖子：`./play_0001_xiaobai.sh --track-neck --fps 60`。包含眼球水平跟踪及已确认的 ID 32，默认颈部 ±160 计数；ID 34 是侧弯，不参与。颈部单独平滑、启停死区和加减速，最大速度 25 计数/秒；最后约 9.1 秒缓慢回中。先只检查可加 `--check`。此模式需先启动同一个视觉服务，不另占摄像头。

颈部加速对比：加 `--neck-speed-multiplier 2`，将 ID 32 的限速设为 50 计数/秒，响应增益和软件加减速限制同步加倍；眼球、歌曲时钟和 ±160 行程不变。此档末尾回中约需 5.9 秒。默认倍率仍为 1，确认效果后按需使用 2。

三倍速度：`./play_0001_xiaobai.sh --track-neck --fps 60 --neck-speed-multiplier 3`。按现场授权，ID 32 软件速度上限已提升为 75 计数/秒；此档响应增益为 180，加减速仍保持两倍档的 70 计数/秒²，平滑、眼球参数及 ±160 范围不变。末尾约 5.2 秒回中；默认仍为一倍。

单颈轴联动验证：`./play_0001_xiaobai.sh --neck-follow-trial 32 --fps 60 --max-seconds 30`。眼球先响应，偏转超出舒适区后颈轴等待约 80ms 再逐步接替；无需等眼球到限位。颈轴默认中位 ±20 计数、最多 25 计数/秒，末尾平滑回中。ID 34 已现场确认为侧弯，当前禁止用于水平跟踪；ID 32 已确认为左右转头轴，编码器正负对应方向仍待确认，需要反向时加 `--neck-follow-sign -1`。该模式不更改默认眼球跟随配置。

若 ±20 计数肉眼不明显，可显式加 `--neck-follow-offset-counts 80 --max-seconds 40` 作逐步观察。此档核对当前设备限位，保留低速，末尾自动延长回中时间至至少 4.2 秒。ID 32、34 均通过适配器的 RS485 接口通信；编码器变化不能替代头部实际运动验收。

独立检查 ID 32 能否双侧转动：`.venv/bin/python tools/xiaobai_neck_sweep.py --play`。不播放歌曲、不指令眼球，仅慢速运行中位→正向80→中位→负向80→中位，每个位姿停留3秒；去掉 `--play` 时仅读检查。

现场要求更大观察幅度时，本机已支持显式 `--offset-counts 160`，对应目标 1887～2207；其他速度与停留参数不变，每次执行前核对当前设备范围。

第一次核对方向可运行 `./play_0001_xiaobai.sh --gaze-direction-trial --max-seconds 20`：仅眼球改为中位 ±40 计数、限速 60 计数/秒，其他口型/表情保持原歌曲幅度。该入口保留“方向待确认”状态，最长试播 30 秒。

按 `Ctrl+C` 停止，或在另一终端执行 `touch /home/ubuntu/project/outputs/xiaobai/STOP`。停止发送不会自动卸掉颈部扭矩。脚本保留 STOP 文件，确认可以重启后由操作员移除。

## 启动条件与日志

舵机电源已开启、TTL/RS485 已连接、音箱已连接；颈部已归位并使能。脚本核对 USB 串号、驱动、读写权限、输入时长、标定哈希及颈部保持状态，不会自动重新使能或抬起未准备好的头部。

USB 拔插后若端口/权限改变，运行：

```bash
sudo python3 /home/ubuntu/project/tools/xiaobai_switch_usb_driver.py vendor
```

重新上电导致未使能或脖子下垂时，先托住并完成已有归位/面部启动流程，再播放；不要靠播放脚本强行恢复。

每次运行创建独立的 `outputs/xiaobai/0001_direct_时间戳/`，保存配置、计划、逐帧日志和串口追踪；反馈异常仅记录，原始追踪有容量上限，完整性以追踪内的丢弃事件数为准。稳定配置存于 `xiaobai/playback_profile_0001_observation.json`。
