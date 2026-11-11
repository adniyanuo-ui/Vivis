# SeetaFace6本地评估来源

- 上游源码：https://github.com/SeetaFace6Open/index
- 官方说明及模型链接：https://github.com/seetafaceengine/SeetaFace6
- 质量评估接口：https://github.com/seetafaceengine/SeetaFaceTutorial
- 固定版本及模型SHA-256见`sources.json`。下载包沿用上游`sf3.0_models`目录名称，不能据此认为使用SeetaFace2。
- 上游许可证原文保存在`SEETA_LICENSE`，各子模块自己的版权及许可证仍保留在隔离源码目录；分发二进制前应保留对应模块通知。
- `probe.cpp`和`quality_test.cpp`为本项目的离线桥接及回归代码，不包含模型权重。
- 后续生产桥接增加了同一YuNet ROI的识别入口和可选年龄模型；不加载性别推断模型。年龄数据是带估计标记的本地观察，不是身份或真实年龄资料。
- `gcc13.patch`只补齐标准库头文件；`brightness_roi.patch`修正固定上游版本的亮度ROI和未初始化图像块使用。报告须明确该修复，不能将修复后的质量表现归为原版SDK。
- 全部构建和模型文件放在`outputs/face_ab`，原始人脸素材、特征和实验明细放在`private/face_ab`；这些目录均被Git忽略。
