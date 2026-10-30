# 墨甲仿生灵巧脸：安装与播放

拿到小盒子（NVIDIA Jetson AGX Orin）后，首次建议接上显示器、键盘和鼠标，便于查看本机IP和摄像头人脸跟踪画面。将仿生灵巧脸自带的USB串口、标记为“Type-B／右”的眼内摄像头接到小盒子，再连接一个USB音箱。确认头颈可靠支撑、外设连接完成后，给小盒子、机器头和音箱上电。

**随仿生头寄出的小盒子已完成第1～3步初始化，用户可以直接跳到第4步播放。** 自备或重新安装的Orin按第1～3步准备。播放已有作品只需阅读本页；播放自己的JSON与音频，继续阅读[新作品快速上手](docs/developer/QUICKSTART.md)。

## 1. 登录并进入源码目录

在小盒子的本机终端操作，或从同一网络的电脑通过SSH登录：

```bash
ssh james@192.168.31.172
```

随箱登录用户名为`james`，密码为`james123`。`192.168.31.172`是当前配置的IP；更换网络后，以小盒子实际IP为准。

源码包已放在`/home/james/project/mojia20261029/`。首次解压：

```bash
cd /home/james/project/mojia20261029
tar -xzf Vivis-mojia-f3d7bd9.tar.gz
cd Vivis
```

已初始化的小盒子无需再次解压，直接进入`/home/james/project/mojia20261029/Vivis/`。

## 2. 安装依赖与离线检查

检查Jetson Orin环境：

```bash
cat /etc/os-release
python3 --version
uname -m
```

已验证环境为 **Ubuntu24.04、Python3.12、ARM64 Jetson AGX Orin**。

安装系统依赖和项目依赖，需要联网：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv libportaudio2 alsa-utils build-essential cmake

./tools/setup.sh
```

安装完成后执行离线检查：

```bash
./tools/check_delivery.sh
```

当前版本预期通过264项测试，最后显示“离线交付自检通过”。这里的“离线”指不连接、驱动机器头；安装依赖仍需联网。运行时使用项目自己的`.venv`，无需激活虚拟环境。

## 3. 核对USB外设并设置串口权限

首次复现使用随寄的墨甲头及配套外设。检查USB设备、串口、相机和音箱：

```bash
lsusb
ls -l /dev/ttyCH343USB0  # 机器头自带的USB串口
ls -l /dev/v4l/by-id/   # 机器头自带的眼内USB相机
aplay -l               # 额外连接的USB音箱
```

首次使用时，需要为串口节点（如`/dev/ttyUSB*`或`/dev/ttyCH343*`）设置访问权限，否则可能报`Permission denied`。随包规则匹配当前配套适配器（`lsusb`中为`1a86:55d3`）：

```bash
sudo install -m 644 tools/99-vivis-servo-access.rules \
  /etc/udev/rules.d/99-vivis-servo-access.rules

sudo usermod -aG dialout "$USER"
sudo udevadm control --reload-rules
```

随后注销并重新登录，重新插拔USB适配器。若串口节点根本不存在，先检查USB连接和设备识别；权限设置不会创建不存在的设备。

配套设备的配置已保存。更换外设时，串口和音频设备名在`configs/robots/mojia/hardware.json`的`runtime`中，相机节点在`configs/robots/mojia/robot.json`的`cameras.right.device`中。更换其他型号音箱时，还应将`robot.json`的`defaults.usb_volume_times_three`设为`false`，使用实际声卡的音量设置。

## 4. 播放示例

外设接好、机器头与音箱上电后，在小盒子终端执行：

```bash
cd ~/project/mojia20261029/Vivis/
sudo systemctl stop dms.service  # 停止可能占用串口的自启动服务
```

若提示`ies.service`占用相机，再执行`sudo systemctl stop ies.service`。以下两条指令逐条运行，上一条结束后再运行下一条。

**播放作品1：**

```bash
./examples/03_face_eye_neck_tracking/run.sh \
  --work 0001 --repeat 1 --parallel-neck on \
  --mouth-command-mode lips-direct \
  --mouth-speed 1200 --round-lip-gain 1.2
```

**播放作品2：**

```bash
./examples/03_face_eye_neck_tracking/run.sh \
  --work wz2 --repeat 1 --parallel-neck on \
  --mouth-command-mode lips-direct \
  --mouth-speed 1200 --round-lip-gain 1.2
```

每遍会独立准备、35轴缓慢回中，再从头播放完整音频与表情。若偶发`RuntimeError: IDxx: home did not settle at neutral`，说明本次回中自检未通过；确认连接与机构正常后，重新运行同一条指令即可。若持续出现，请保留报错中的ID和日志目录供排查。

人脸跟踪运行时，在 **Orin本机浏览器** 打开：

```text
http://127.0.0.1:8765/
```

可查看摄像头跟踪画面。检查声音与面部表情、眼球跟随，以及颈部左右、俯仰和轻微侧倾跟随。SSH电脑上的`127.0.0.1`指向电脑自身；本说明使用Orin本机浏览器。

### 播放参数说明

| 项目 | 含义 |
|---|---|
| `examples/03_face_eye_neck_tracking/run.sh` | 音频与面部表情，加横向眼球和颈部人脸跟踪；默认使用右眼摄像头。 |
| `--work 0001`／`--work wz2` | 选择已登记的作品名，同时加载对应音频与表情轨迹；不是直接传JSON路径。 |
| `--repeat 1` | 完整播放1遍；改为2即播放2遍，每遍均独立准备和回中。 |
| `--parallel-neck on` | 开启ID33/34并联平台的俯仰和轻微侧倾；ID32左右转由示例3原有跟踪负责。 |
| `--mouth-command-mode lips-direct` | 歌曲阶段将10个圆唇相关轴及ID31下颌的映射目标直接交给发送器，不再经过该组的软件逐帧限速；舵机硬件限速、启动避让和回中仍保留。 |
| `--mouth-speed 1200` | 本次任务将14个口型相关轴的硬件运行速度设为1200计数/秒；按原启动流程写入、读回SRAM，不改永久配置文件。 |
| `--round-lip-gain 1.2` | 仅将`MouthFunnel`／`MouthPucker`两个圆口、嘟唇通道乘1.2后按标定映射，权重最高为1；不改变其他表情。省略时为1.0。 |

需要只检查、不播放时，在播放指令末尾加`--check`。正常播放结束显示`ROUND 1/1 COMPLETE`。运行中按`Ctrl+C`停止音频和新位置指令，程序清理本次资源并保留颈部扭矩。

终端会打印本次日志目录，位于`outputs/developer/`下。播放新的作品、选择仅面部或增加眼球跟踪，见[新作品快速上手](docs/developer/QUICKSTART.md)。
