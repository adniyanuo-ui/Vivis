# 歌曲节目单与离线制作

## 1. 点播和退出

交互态直接说完整歌名，例如「唱一首 You're Beautiful」「播放李健的给你」「播放 golden hour」。多个歌曲都可用时，只说「唱首歌」会询问具体歌曲，不再固定选择白骨夫人。

说「停止」「退出」回交互态；说「待机」回接待待机。节目播放完毕自然回交互态。歌曲沿用静态表演流程：准备、自检回中、完整音频与表情、眼颈跟随；不叠加实时微笑回应。

手动测试某首歌（先停止占用设备的交互服务）：

```bash
sudo systemctl stop vivis-p1.service
./examples/03_face_eye_neck_tracking/run.sh \
  --robot configs/robots/xiaoshuai/robot.json \
  --work jvke_golden_hour --repeat 1
```

测试后恢复交互服务：`sudo systemctl start vivis-p1.service`。

## 2. 素材和配置来源

- [收录清单](../../assets/program_requests/songs_20261112.csv)：歌曲、来源网址、原音频哈希、用户采用状态。
- [节目单](../../configs/interaction/programs.json)：可点播的作品及别名。只有完整制作且通过检查的作品才能进入。
- [小帅作品清单](../../configs/robots/xiaoshuai/works.json)：音频、61通道源动画、标定后舵机目标及哈希，不复制运动参数。
- `assets/works/songs/<work_id>/`：播放音频、源动画、制作来源记录。
- `assets/works/xiaoshuai/<work_id>/`：使用小帅当前标定生成的目标位置。
- `assets/works/mojia/<work_id>/`：同一源表情对应的墨甲目标位置；两种本体共用音频，不复制各自硬件设置。
- `incoming/songs/`：原始获取素材，保持不改；`outputs/song_build/`：可重新生成的分离人声、模型缓存和推理日志。

原曲完整混音转换为24kHz单声道PCM16用于播放，不剪切、不改变时间轴、不额外增益。单独分离的人声只用于生成口型，音箱仍播放含伴奏的完整混音。原始下载版本及哈希保留，不能将短试听片段当成完整作品。

## 3. 离线制作链路

原曲 → UVR Voc_FT人声分离 → 现有本地A2F → 按名称适配52个基础表情 → 以音频采样时间插值到60FPS → 当前机器人标定与硬限位映射 → 静态播放目标。

另外9个视线/姿态通道为零，实际眼颈由既有跟随控制拥有。源表情增益1.0；本轮不增加闭嘴修饰、额外软件限速或改变设备参数。静态动画生成不等于真机皮肤效果验收，实际口型仍需现场观察。

制作工具为[prepare_song_library.py](../../tools/prepare_song_library.py)、[separate_song_vocals.py](../../tools/separate_song_vocals.py)，均不打开音箱、相机或舵机。它们需要额外离线模型环境；日常播放不依赖人声分离模型或这些工具。

分离模型来源：[UVR模型发布页](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models)，使用`UVR-MDX-NET-Voc_FT.onnx`；参数参考[UVR模型元数据](https://github.com/TRvlvr/application_data/blob/main/mdx_model_data/model_data.json)。其SHA256为`534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111`。模型和TensorRT缓存仅保留在本机`outputs/song_build/models/`，不改变生产Python依赖。

```bash
# 仅首次在当前Orin上构建分离引擎。
trtexec --onnx=outputs/song_build/models/UVR-MDX-NET-Voc_FT.onnx \
  --saveEngine=outputs/song_build/models/vocals_fp16.plan \
  --fp16 --shapes=input:1x4x3072x256 --skipInference

# 制作清单中已明确采用且音频哈希匹配的作品。
PYTHONPATH=src .venv/bin/python tools/prepare_song_library.py \
  --engine outputs/song_build/models/vocals_fp16.plan

# 所有完整素材校验后，将ready作品登记为可点播节目。
PYTHONPATH=src .venv/bin/python tools/prepare_song_library.py \
  --engine outputs/song_build/models/vocals_fp16.plan --register-only
```

离线工具可复用已完成的人声分离和A2F结果，但不会跳过音频哈希核对。失败的原生推理目录需先检查日志，再移走后重试。`awaiting_audio`条目仅为待办，不向用户承诺可播放。

## 4. 本批次曲目（2026-11-12）

| 歌手 | 歌曲 | `--work` | 完整时长 |
|---|---|---|---|
| James Blunt | You're Beautiful | `james_blunt_youre_beautiful` | 3:22 |
| James Blunt | 1973 | `james_blunt_1973` | 4:03 |
| James Blunt | Carry You Home | `james_blunt_carry_you_home` | 3:52 |
| James Blunt | Same Mistake | `james_blunt_same_mistake` | 5:03 |
| James Blunt | Bonfire Heart | `james_blunt_bonfire_heart` | 3:46 |
| 李健 | 给你 | `li_jian_gei_ni` | 4:18 |
| 李健 | 贝加尔湖畔 | `li_jian_beijiaer_hupan` | 4:07 |
| 李健 | 传奇 | `li_jian_chuan_qi` | 4:53 |
| 李健 | 异乡人 | `li_jian_yi_xiang_ren` | 9:46 |
| 李健 | 似水流年 | `li_jian_si_shui_liu_nian` | 3:37 |
| 李健 | 假如爱有天意 | `li_jian_jia_ru_ai_you_tian_yi` | 5:04 |
| 周杰伦 | 稻香 | `jay_chou_dao_xiang` | 3:43 |
| JVKE | golden hour | `jvke_golden_hour` | 3:33 |
| 余文乐 / 艾热 AIR / 王以太 | 哀人i (Live) | `ai_ren_i_live` | 4:12 |

用户试听采用前13首，随后指定加入《哀人i (Live)》。保留采用的现场／试听设备演示来源版本，不冒充录音室母带。

凤凰传奇《山河图》（网易云ID1350658663）已登记为待补素材：公开音频入口返回HTML页面，未取得可解码的完整音频，因此不出现在可点播节目单中。需要提供完整MP3/WAV等音频后再制作。此前其余候选曲仍以收录CSV状态为准。
