# 可选AEC3人声保护扩展

该扩展只用于隔离实时语音试验，不改变P1或原始`pywebrtc-audio`包。通过构造参数设置AEC3的双讲判定阈值，替代离线实验模块的环境变量；不包含线性输出旁路、任意环境变量调参或设备专用代码。

`sources.json`固定源码提交、原始文件摘要、补丁摘要和构建依赖。`profile.patch`只修改绑定层与模块构建名称：

1. 新模块`_vivis_aec3`仅暴露独立的处理器类型，可与原模块共存。
2. 构造参数`nearend_threshold`限制为有限的0.25～2.0，重置后保留。
3. Vivis中的`voice-preserve`配置传入已试听的1.0；`baseline`继续使用原包和默认0.25。

源码和本地二进制位于被忽略的`outputs/speech/`，不进入Git。运行时从独立语音环境的site-packages导入，不引用诊断目录或另一份仓库。安装命令：

```bash
./tools/setup_realtime_voice.sh --voice-preserve
```

需要系统已有Git、patch、C++编译器、CMake及Python开发头文件。构建独立安装固定pybind11；不写系统Python、固件或全局音频设置。构建信息与二进制SHA256保存在该环境的`_vivis_aec3_build.json`，编译日志和上游许可／NOTICE保留在构建目录。

补丁基于[pywebrtc-audio](https://github.com/strands-labs/pywebrtc-audio)的Apache-2.0封装，WebRTC及第三方代码沿用上游各自许可和声明。保留完整上游来源树；再分发编译产物时须同时核对并附带所用组件的许可，不能把封装许可当作全部组件许可。本仓库不分发该二进制。

较宽松的双讲判定可能保留更多用户声音，也可能增加回声残留。该候选仅有单段离线试听改善，实时声学与云端验收须单独记录，不宣称完全无损。
