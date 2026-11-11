# SeetaFace6与SFace本地对照

## 1. 目的和当前边界

在Jetson AGX Orin上验证SeetaFace6能否改善当前背光场景的熟人识别，区分检测、特征比对、质量拒绝和业务确认策略的影响。现有P1继续使用SFace及已完成的3次确认／5秒复核，测试工具不替换模型、不更新登记库、不控制舵机，也不上传照片、视频或特征。

目前只有6张带框的现场截图，没有找到原始现场视频。截图可用于冒烟测试，不能当作正式视频，也不能据此宣称时序稳定或真实串人率为零。正式结论须等待下面的人员配合。

## 2. 编译、模型及质量评估

官方源码：[SeetaFace6Open/index](https://github.com/SeetaFace6Open/index)。官方使用说明：[SeetaFace6教程](https://github.com/seetafaceengine/SeetaFaceTutorial)。固定提交、子模块和模型哈希见[来源清单](../../native/vision/seeta/sources.json)，许可证见[原文](../../native/vision/seeta/SEETA_LICENSE)。

本机已采用CPU／ARM64 NEON编译。源码、二进制、模型和日志均在被忽略的`outputs/face_ab/`下，没有安装到系统目录，也没有改变CUDA或生产环境依赖。复现：

```bash
cd ~/project/Vivis-delivery
.venv/bin/python tools/setup_seetaface6.py --jobs 4
```

`setup.log`是完整编译日志，`build_manifest.json`记录源码、模型与补丁哈希。适配只含明确的问题修复：

1. [GCC 13兼容补丁](../../native/vision/seeta/gcc13.patch)：补齐`<functional>`和`<cstdlib>`，不放宽编译类型检查。
2. [亮度ROI修复](../../native/vision/seeta/brightness_roi.patch)：该固定版`QualityOfBrightness`把未初始化图像块`fill`到画面副本，且随后评估整幅画面，导致同一暗图重复检测的亮度等级不一致。改为裁剪实际人脸矩形、使用正确宽高，并对人脸区域评估。**质量过滤结果必须注明使用此补丁，不能当成未经修改的官方版本。**

质量部分使用SDK自带的`QualityOfBrightness`和`QualityOfClarity`传统评估器，保持默认分级：任一为LOW就在特征提取前拒绝。没有宣称使用深度`quality_lbn`；后者需要68点和额外模型，本轮未接入。质量过滤不能恢复缺失细节，也不能挽回检测器已经漏掉的人脸。

新增原生回归使用合成亮／暗人脸块，在明亮背景下重复100次，检查ROI等级、结果确定性和输入不变。实际正常登记照可提取特征、自比对约1；同图压暗后质量LOW，重复3次均跳过特征提取，识别阶段耗时为0。

## 3. 四组对照与指标定义

| 组 | 检测与识别 | 用途 |
|---|---|---|
| `sface320` | 当前生产YuNet 320宽＋SFace | 真实基线；直接读取现有登记特征 |
| `sface640` | YuNet 640宽＋相同SFace登记特征 | 分离检测输入分辨率影响 |
| `seeta6` | Seeta检测＋5点＋1024维识别 | 观察模型自身表现；质量只记录、不拦截 |
| `seeta6_quality` | 相同Seeta模型＋LOW质量前置拒绝 | 检查过滤收益和拒识代价 |

两种模型采用相同五人的登记照片。SFace读取现有`private/people/gallery.npz`，不重新登记；Seeta另行提取其自身特征，仅用于当前测试。Seeta登记图采用现有登记流程相同的最长边1280预处理。模型特征维度和分数标度不同，不能混用特征、比较分数绝对大小或机械使用同一个阈值。

初始实验阈值：SFace保留0.5／候选差距0.08；Seeta暂用0.6／0.08作为待验证工程初值，并非现场标定结果。正式调阈值时必须分开开发集和最终评估集，不能用评估素材反复试到最好后再报告。

每段视频按真实采集时间约2Hz采样，四组使用同一帧；按采样序号轮换运行顺序，单独记录各组耗时。OpenCV和Seeta可配置的检测／识别线程均设1，Seeta关键点模块使用SDK默认行为。Seeta登记阶段已运行推理；模型加载不计入单帧耗时，OpenCV首次网络执行或输入形状变化仍可能影响首帧，`--repeats 3`取每帧3次耗时中位数降低首轮偏差。冷启动响应不是该指标覆盖的内容。

| 指标 | 分母／边界 |
|---|---|
| 检测覆盖率 | 人工约定有人脸的帧中，检测到至少一个人脸的比例；未做真实框IoU标注，不能当作完整检测精确率／召回率 |
| 已知人拒识率 | 已知人员单人帧中输出未知的比例；包含漏检、质量拒绝和分数不足，不能把过滤帧从分母删掉 |
| 已知人串人率 | 已知人员帧中输出其他登记ID的比例 |
| 陌生人误认率 | 未登记人员单人帧中输出任一登记ID的比例；没有陌生人素材时为N/A |
| 身份反复 | **确认／复核层之前**相邻采样帧的ID／未知变化次数；不同录像之间不计，截图为N/A。用于看模型输出抖动，不等于P1经过保持策略后的体验 |
| 分数分布 | 正确人员分数、其他人员分数的P10／P50／P90／P95／最大值；质量过滤组只对实际提取的帧有分数，须结合未过滤组判断 |
| 单帧耗时 | P50／P95／最大值；SFace完整调用包含现有预览JPEG生成，Seeta原生调用没有预览，另外有原生检测／关键点／质量／识别分阶段耗时，不能把两种总耗时当作纯网络推理耗时 |

只读`frames.jsonl`可以逐帧排查失败原因；`report.json`保存计数、分母、输入／登记库哈希及分布；`report.md`给同事查看。所有原始材料和带人员ID的明细保留本机`private/face_ab/`，不加入Git。

## 4. 现场配合：先录制，后离线运行

建议何工、小王各自单独录三段：正常光照正脸20秒、当前背光位置正脸20秒、轻微左右转头20秒。另请一位未登记同事在正常／背光下各录20秒；可以追加无人画面。不要多人同时站在镜头里，也不要在一段录像里换人。使用`--preview`先在Orin浏览器打开`http://127.0.0.1:8765/`，确认额头、双眼和下巴完整入镜后回终端按Enter，才开始5秒倒计时。取景蓝框只画在预览副本上，不写入原始视频。每段开始后持续保持脸在画面里；若离开画面，需在清单里裁去对应时间，而不是错误标为拒识。

先停止交互服务释放相机，录制只打开相机，不采麦克风、不发送舵机指令：

```bash
cd ~/project/Vivis-delivery
sudo systemctl stop vivis-p1.service

# 何工，看到提示后5秒开始；每段20秒。
.venv/bin/python tools/record_face_ab.py --preview --person heyude --scene normal
.venv/bin/python tools/record_face_ab.py --preview --person heyude --scene backlight
.venv/bin/python tools/record_face_ab.py --preview --person heyude --scene turning

# 小王，同样三段。
.venv/bin/python tools/record_face_ab.py --preview --person wangrunzhang --scene normal
.venv/bin/python tools/record_face_ab.py --preview --person wangrunzhang --scene backlight
.venv/bin/python tools/record_face_ab.py --preview --person wangrunzhang --scene turning

# 未登记同事，不能用已登记人员冒充陌生人。
.venv/bin/python tools/record_face_ab.py --preview --person unknown --scene normal
.venv/bin/python tools/record_face_ab.py --preview --person unknown --scene backlight

```

视频采用FFV1无损编码，不画框、不叠文字；相机原有180度方向校正保留。保存逐帧真实采集时间，避免用容器名义30FPS掩盖USB掉帧。摄像头本身输出MJPEG的原有压缩仍存在；“无损”是指不再增加录制编码损失。8项Python回归包含FFV1逐像素还原、真实时间采样、生产60像素门槛及质量拒绝的计数口径。

每次完成会打印一个具体目录。将**本轮明确录好的目录**传给清单工具，避免用宽泛通配符混入以前录像：

```bash
# 下列“目录1、目录2”必须换成本轮打印出的实际目录。
.venv/bin/python tools/make_face_ab_manifest.py \
  --capture 目录1 目录2 \
  --output private/face_ab/现场对照.json

.venv/bin/python tools/benchmark_faces.py \
  --manifest private/face_ab/现场对照.json \
  --output private/face_ab/现场对照结果 --repeats 3

# 比对期间先保持交互服务停止，避免后台视觉推理影响耗时；完成后恢复。
sudo systemctl start vivis-p1.service
```

这两个输出路径要求尚未存在，避免覆盖已有实验。只需要录一次，后续四组全部使用这些文件，不重复请人表演同样动作。

## 5. 决策标准和尚未完成的工作

先看同等测试集上的串人和陌生人误认，再看拒识与反复；质量过滤不能靠“全部拒绝”获胜。若Seeta在背光下检测已经较差，先记录该事实，不能用质量模块掩盖。2Hz应用单次总处理预算目标为500ms，是否满足由正式录像P95及并发负载实测判断。

有限样本的零次误认只报告0/N，不宣称真实误认概率为零。只有正式多人的原始录像验证有明确收益，才讨论可选后端、质量门控如何接入现有复核层，以及是否替换SFace；目前不切换生产配置。

现有6张带框截图的预检查中，两种模型都没有通过既定比对门槛，Seeta没有检测到人脸。由于素材有叠图、没有连续时序、只有一人且没有陌生人，这只能提示必须认真做原始录像试验，**不支持“Seeta一定更好”或“SFace已经足够”的结论**。

## 6. 本轮已完成的验证（2026-11-11）

- Orin本机原生ARM64构建成功，重复执行构建脚本通过；检测、5点、1024维识别、相似度计算与亮度／清晰度质量评估可调用。
- 以登记照做端到端自回测，四组均能返回正确ID。这是链路冒烟测试，不是独立识别精度评估。
- 对现有五人的原始照片按原登记流程重新提取SFace特征，分别与保存库比对，余弦均约1，确认测试使用的源照片与生产底库一致；没有写回或替换底库。
- 质量ROI的100次合成图回归通过；压暗登记照连续3次稳定为LOW，均跳过特征提取。
- 隔离源码、共用已安装Python环境的交付自检686项，683项通过、3项跳过；包含8项新增评估／文件录制格式回归。`git diff --check`通过。
- 尚未录制正式无叠图多人视频，尚未计算可用于选型的现场拒识／串人／时序稳定性结论。没有切换P1识别模型或启动新的真实相机录制。
