# 可选本地熟人识别模型来源

当前P1默认使用SeetaFace6作身份比对、质量评估及可选年龄观察，YuNet继续负责轻量检测与跟随。见[运行说明](../../docs/developer/SEETAFACE6_RUNTIME.md)。以下SFace来源保留作为显式回退与A/B基线。

SFace通过现有YuNet的五点关键点对齐图像并提取特征；运行于本机，不向外部服务上传照片。本模块不控制舵机。

- 官方实现：[OpenCV人脸检测与识别教程](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html)。
- 官方模型目录：[opencv_zoo / face_recognition_sface](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface)。
- 固定版本、下载地址、文件大小及SHA-256：[sface_sources.json](sface_sources.json)。
- 该模型目录的许可证原文：[SFACE_LICENSE](SFACE_LICENSE)。

执行`python3 tools/setup_familiar_faces.py`下载到被忽略的`outputs/models/sface/`。模型不强制随核心项目安装，不把本机个人特征库放进公共素材目录。

当前相似度门槛与连续确认参数是本项目可配置的初值。官方教程的示例阈值不等于本机、当前相机和登记照片的现场准确率；后者需要单独验收。

SeetaFace6的ARM64本地构建、源码及质量ROI补丁见[seeta来源说明](seeta/NOTICE.md)，现场原始录像与四组对照步骤见[SeetaFace6 A/B测试](../../docs/developer/SEETAFACE6_AB_TEST.md)。它目前是独立评估工具，没有切换生产识别后端。
