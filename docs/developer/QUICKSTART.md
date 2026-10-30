# 用自己的JSON和音频播放新作品

先按[README](../../README.md)完成安装、连接和随包作品播放。本页只说明如何导入、检查和播放自己的作品；所有命令在`Vivis/`源码目录执行。

## 1. 准备一份表情JSON和对应音频

例如将两个文件放到：

```text
/home/james/project/mojia20261029/incoming/my_demo/
├── my_demo.json
└── my_demo.wav
```

这里的JSON是逐帧表情权重序列，不是机器头标定JSON。文件名可以自定，导入时填写实际路径。

| 输入 | 要求 |
|---|---|
| 表情JSON | 顶层只包含`name`、`fps`、`blendShapes`三个字段；每帧为61个有限数值。 |
| 帧率 | 支持30FPS或60FPS，如`"fps": {"num": 30, "den": 1}`。 |
| 通道 | 顺序与`configs/schemas/blendshape_channels_61.csv`中的61个通道一致；CSV前两列`Timecode`和`BlendshapeCount`不放进帧数组。前52维为0～1表情权重，后9维为头部和双眼旋转（弧度）；不使用的通道也要填0，不能省略。 |
| 音频 | 16位PCM编码的WAV文件；MP3、MP4等应先转换为该格式。 |
| 时间对齐 | 音频与表情从同一时刻开始，JSON总帧数÷帧率应与音频时长一致，误差不超过一个源帧（30FPS约33.3毫秒，60FPS约16.7毫秒）。 |

JSON格式如下。这里只展示一帧61个零值的中性表情，实际作品需包含与音频对应的完整逐帧数据：

```json
{"name": "my_demo", "fps": {"num": 30, "den": 1}, "blendShapes": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

当前三个示例中，眼球和颈部是否参与跟踪由示例模式决定；示例3的颈部姿态来自实时人脸跟踪，不播放JSON中的头部旋转通道。

## 2. 导入作品

```bash
cd ~/project/mojia20261029/Vivis/

PYTHONPATH=src .venv/bin/python -m vivis.motion.export \
  --name my_demo \
  --clip "$HOME/project/mojia20261029/incoming/my_demo/my_demo.json" \
  --audio "$HOME/project/mojia20261029/incoming/my_demo/my_demo.wav"
```

| 参数 | 含义 |
|---|---|
| `--name my_demo` | 注册后的作品名，播放时用`--work my_demo`选择。仅用英文字母、数字、下划线或连字符，首位为字母或数字，长度1～64；不能与已有作品重名。 |
| `--clip` | 原始61通道表情JSON的路径；路径有空格时保留双引号。 |
| `--audio` | 对应PCM16 WAV音频的路径。 |

导入是离线处理，不驱动机器头，也不修改原始JSON和音频。工具按随箱标定和限位生成电机目标，将源文件副本、音频和目标保存到`outputs/imported_works/my_demo/`，并自动在`assets/works.json`登记新作品。30FPS源数据每帧保持两次生成60FPS轨迹；60FPS源保持原帧率。

## 3. 先检查，再播放

先核验新作品及示例3的运行参数，不打开硬件：

```bash
./examples/03_face_eye_neck_tracking/run.sh \
  --work my_demo --repeat 1 --parallel-neck on \
  --mouth-command-mode lips-direct --mouth-speed 1200 \
  --check
```

通过后，确认机器头与音箱上电、串口与右眼相机已连接，并按README停止占用设备的服务，再运行：

```bash
./examples/03_face_eye_neck_tracking/run.sh \
  --work my_demo --repeat 1 --parallel-neck on \
  --mouth-command-mode lips-direct --mouth-speed 1200
```

**新导入作品先不加`--round-lip-gain 1.2`，使用默认1.0。** 随包`0001`和`wz2`已登记该功能需要的源文件信息；当前导入工具虽保存源文件副本，但不会登记运行时圆唇增益所需的字段，直接照搬1.2会报`This work has no verified source clip for runtime channel gain`。这不影响按原始表情信号播放新作品。

需要简化运动组合时，改用以下入口；上一条运行结束后再执行下一条：

```bash
# 示例1：音频＋面部表情，不启动相机；播放时眼球和颈部保持
./examples/01_face_control/run.sh \
  --work my_demo --repeat 1 \
  --mouth-command-mode lips-direct --mouth-speed 1200

# 示例2：音频＋面部表情＋横向眼球跟踪；播放时颈部保持
./examples/02_face_eye_tracking/run.sh \
  --work my_demo --repeat 1 \
  --mouth-command-mode lips-direct --mouth-speed 1200
```

三个示例每遍均先执行35轴准备与缓慢回中，“保持”指正式播放阶段。示例2、3的摄像头画面在Orin本机`http://127.0.0.1:8765/`查看。播放次数用`--repeat`调整，其余参数含义见README。

正常完成会显示`ROUND 1/1 COMPLETE`，本次日志目录中的`performance/summary.json`应为`"completed": true`。用`Ctrl+C`停止时会清理资源、保留颈部扭矩，停止发送不等于切断舵机电源。

## 4. 保存、修改与转移新作品

- 已导入的作品下次直接用`--work my_demo`播放，无需再次导入。
- 修改了源JSON或音频后，换一个作品名（例如`my_demo_v2`）重新导入；工具拒绝覆盖同名作品。
- 保留`outputs/imported_works/my_demo/`整目录及对应的`assets/works.json`。导入素材位于被Git忽略的`outputs/`下，仅推送代码不会自动带走它；不要将整个`outputs/`当作可随意清空的日志目录。
- 转移到同版本、同标定的另一台设备时，同时复制作品目录和更新后的清单，保持相对路径不变，再运行一次`--check`。

## 5. 常见问题

| 提示或现象 | 处理 |
|---|---|
| `Unexpected clip fields`／`Invalid clip array` | 检查JSON顶层字段、每帧61个数值、通道顺序及0～1表情权重。 |
| `Need PCM16 WAV matching the clip duration within one source frame` | 检查音频编码、JSON帧率、帧数和音频时长是否匹配。 |
| `Work already exists; use a new name` | 更换`--name`，例如`my_demo_v2`。 |
| `Unknown work` | 先完成导入，播放时填写导入用的作品名，而非文件名或路径。 |
| `no verified source clip for runtime channel gain` | 新导入作品省略`--round-lip-gain 1.2`，保留默认1.0。 |
| 串口缺失、权限被拒绝、相机或音箱打不开 | 按README第3、4步检查设备、权限和占用服务；保留完整报错及终端打印的日志目录。 |
