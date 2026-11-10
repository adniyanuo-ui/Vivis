# 小帅P1密钥保存与开机自启动

当前默认方案（2026-11-10）：语音理解与节目控制已迁移到Qwen实时工具调用；前台接待待机保留相机观察，来人主动问候，并启用按人记忆。本文旧版本地ASR/关键词、固定提示音及待机关闭相机的描述属于历史方案；以[Qwen前台接待与记忆](QWEN_RECEPTION_AND_MEMORY.md)及当前配置为准。

## 1. 只输入一次密钥

在小盒子本机，以运行程序的普通用户执行：

```bash
cd ~/project/Vivis-delivery
.venv/bin/python tools/configure_api_key.py
```

按提示输入北京地域的百炼API Key。输入不回显，保存为`~/.config/vivis/dashscope_api_key`，权限600，仅本机使用，不放进代码仓库、命令行参数或日志。这里是本机文件保存，区别于`--ask-key`的临时输入。

此后手动启动可直接运行：

```bash
.venv/bin/python -m vivis.interaction.programs dialogue \
  --robot configs/robots/xiaoshuai/robot.json
```

密钥读取顺序：当前进程的`DASHSCOPE_API_KEY`环境变量 → systemd提供的credential → 本机默认密钥文件。如果终端曾export过旧Key，请先执行`unset DASHSCOPE_API_KEY`再用新文件。更新或撤销Key后重启程序或服务，才会重新读取。密钥失效或撤销后仍需更新，不能永久免维护。

## 2. 原命令的参数

| 部分 | 含义 |
|---|---|
| `.venv/bin/python` | 使用本项目虚拟环境中的Python及依赖 |
| `-m vivis.interaction.programs` | 按Python模块启动统一节目入口 |
| `dialogue` | 启动三级状态管理器：真正待机 → 唤醒交互 → 节目任务；持续监听本地控制指令 |
| `--robot configs/robots/xiaoshuai/robot.json` | 使用小帅的标定、硬件、跟随及知识配置引用，不切换到默认墨甲 |
| `--ask-key` | 强制在终端临时询问本次Key，不保存；使用持久文件或开机服务时省略 |
| `--record-audio` | 显式保存原始麦克风、播放参考和处理后音频，供AEC、ASR等排查；不是开启麦克风的开关 |
| 行末`\` | Shell换行续写，将多行组成一条命令；最后一行不写 |

不加`--record-audio`仍能正常聊天、唤醒、停止、唱歌和演剧本，也仍保存状态、转写和诊断日志；只是不会持续保存三份原始调试录音。

## 3. 开机服务

安装与启用（不立刻启动机器头）：

```bash
.venv/bin/python tools/install_autostart.py
```

默认安装当前源码目录、小帅配置及当前普通用户；不要以`sudo python`运行安装器，工具会在安装系统文件时自行调用sudo。生成的服务文件为`/etc/systemd/system/vivis-p1.service`，使用当前目录的绝对路径。移动目录、切换运行用户或更换机器人后，应重新安装对应配置：

```bash
.venv/bin/python tools/install_autostart.py \
  --robot configs/robots/xiaoshuai/robot.json
```

准备好机器头供电、USB串口、相机、阵列麦克风和USB音箱后，先退出当前手动运行的P1，再启动服务：

```bash
sudo systemctl start vivis-p1.service
systemctl status vivis-p1.service --no-pager
journalctl -u vivis-p1.service -f
```

按`Ctrl+C`仅结束日志查看，不会停止后台服务。开机服务无需登录桌面或打开终端，初始进入真正待机，摄像头关闭。喊“小帅”后进入交互态，后续点播、退出、待机沿用现有流程。

服务使用systemd的`LoadCredential`读取本机密钥，启动命令和服务文件不包含Key明文。默认不持续录制调试音频。启动前最多等待120秒检查USB枚举和串口权限，不发送舵机指令；设备就绪后才由原P1执行自检和回中。此检查不能判断机器头电源是否实际开启，必须由操作者确认。

服务与`dms.service`、`ies.service`互斥，避免抢串口或相机。它使用原有唯一控制权锁；不要同时手动启动另一套P1。`systemctl is-enabled`表示已设置开机启动，`active`只表示进程存活，舵机自检通过仍需查看日志中的`SELF-CHECK PASS`及后续状态。

## 4. 停止、调试和恢复

```bash
# 停止服务，沿用原有退出与保持扭矩流程
sudo systemctl stop vivis-p1.service

# 停止后可按原命令手动调试并显式录音
.venv/bin/python -m vivis.interaction.programs dialogue \
  --robot configs/robots/xiaoshuai/robot.json --record-audio

# 退出手动程序后恢复服务
sudo systemctl start vivis-p1.service

# 取消开机启动并停止当前服务
sudo systemctl disable --now vivis-p1.service

# 再次启用下次开机启动
sudo systemctl enable vivis-p1.service
```

若程序或设备故障退出，服务不会自动反复重启，避免故障状态下重复自检和回中。先排查日志、供电和连接，再手动启动。`stop`先通知父管理器，由它有序停止语音、视觉和节目子进程；超过30秒才由systemd清理残留进程。

交互预览仍在Orin本机`http://127.0.0.1:8765/`，仅摄像头开启时可见。详细运行日志仍位于项目`outputs/p1/`。密钥配置与服务安装的离线检查不代表已经验证过真实重启后的供电、网络及设备启动时序。

## 5. 2026-11-09现场启动问题

首次服务启动在设备枚举通过后，被密钥权限检查拒绝：本机systemd255将`LoadCredential`运行时文件创建为私有挂载内的root:root、0440，而原检查只接受用户文件的0600权限规则。已将两种来源分开校验：本机持久文件仍要求0600或更严格；systemd凭据只接受受限只读普通文件，不允许其他用户权限或写权限。未更改密钥内容、原文件权限、舵机配置，也未自动重启机器头。已用非真实密钥在systemd短时测试单元验证读取成功；完整开机运动验收仍待执行。
