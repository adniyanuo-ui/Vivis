# 墨甲开发者快速上手

版本0.1｜2026-10-26｜三个新入口待现场逐例验收。

## 1. 准备环境

机电头、匹配电源、USB串口通信连接、客户自备Linux主机及音频输出。示例2/3需要一只已配置的眼内USB相机，Orin主控可选。需要Python3.10–3.12；本机为3.12。

本体包含35轴和两只眼内相机。USB控制器到TTL/RS485的转接、电平转换及供电应按实际本体线束/控制板连接，不能把不同电平端子直接互连。

## 2. 安装

仓库根目录执行；已有可用`.venv`直接复用，不重装CUDA或驱动。

```bash
sudo apt-get install python3-venv libportaudio2
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[vision]'
python3 tools/bootstrap_sources.py --component servo
.venv/bin/python tools/download_yunet.py
```

只跑面部示例可安装`pip install -e .`，无需OpenCV/YuNet。固定版本SDK恢复工具不安装或切换USB驱动。当前Linux的V4L2、ALSA及进程锁不等于Windows支持。

## 3. 恢复作品

按[素材说明](../../assets/README.md)恢复4个运行文件。从单独提供的素材目录导入缺失文件：

```bash
PYTHONPATH=src .venv/bin/python -m vivis.runtime.assets --from-dir /path/to/supplied-assets
```

当前机器素材齐全时，去掉`--from-dir`只做核验。已有文件不覆盖；哈希不符需定位来源。

## 4. 配置设备

入口为`configs/robots/mojia/robot.json`。核对串口、波特率、音箱和相机节点；示例2/3可用`--camera-device /dev/v4l/by-id/实际节点`指定。串口USB重插后需仍有权限，部署规则见`tools/99-vivis-servo-access.rules`。

默认右眼参考ID23，相机640×480/30FPS、检测目标20Hz、运动60Hz，各频率相互独立。横向双眼1987–2107，颈部32沿设备范围1600–2600。500kbps候选未启用，默认仍1Mbps。启动器只提示占用服务，不自行停服务、重启USB或切驱动。

## 5. 离线检查

```bash
./examples/01_face_control/run.sh --check
./examples/02_face_eye_tracking/run.sh --check
./examples/03_face_eye_neck_tracking/run.sh --check
```

检查素材/标定哈希、目标范围、回中清单、速度及轴控制权；不打开串口，不启动音频或相机。通过不代表设备在线或机械动作已验收。

## 6. 真机验收和停止

核对本体供电、通信及头颈支撑，按三个示例README逐个运行。先wz2，再选择`--work 0001`。每遍独立准备、按需执行既有机构恢复、35轴以速度60回中，恢复任务速度后完整播放。回中会涉及35轴，“保持”指正式播放阶段。

使用[验收表](ACCEPTANCE.md)记录，旧入口效果不直接计作新入口已通过。日志为`outputs/developer/时间_模式/`，`performance/summary.json`记录总体结果，`round_N/preparation.json`和`playback_log.json`记录准备和播放；相机日志为`vision.log`及`vision/summary.json`。

Ctrl+C停止发送和音频，释放接收器、串口及自行启动的相机，保留颈部扭矩，沿用既有舌部卸载策略。舵机可能继续完成最后一个目标；停止发送不等于断电急停。作品结束不按静音停顿改写口型，也不强制面部回中。
