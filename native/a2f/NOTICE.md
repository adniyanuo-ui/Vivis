# A2F原生推理入口

此目录为P0开发中的原生推理入口，不直接打开相机、音频输出或舵机串口。

SDK及依赖来源固定在`sources.json`。默认构建位置是`outputs/a2f/`；可用CMake参数`VIVIS_A2F_WORKSPACE`指定已准备的独立依赖目录。模型和引擎不随源码提交。

`compat_headers/tbtSVD/SVD.h`基于MIT许可的wi-re/tbtSVD提交`21e06aad232e0a58f77d0bd2df3470712529ab33`，许可证随附。仅调整平台头文件及ARM64主机浮点模式保存／恢复：ARM64使用FPCR的FZ位，代替x86 MXCSR的DAZ／FTZ位；GPU上的SVD计算步骤保持。默认SDK依赖包只有x86构建入口，因此依赖采用ARM64原生构建。

CMake在构建目录生成两个SDK CUDA源文件的兼容副本，将显式`thrust::pair`返回值声明改为`auto`以适配CUDA13的CCCL；SDK检出内容保持不变。SDK为NVIDIA MIT许可源码，完整许可证位于下载后的SDK根目录。

`a2f_chunk_probe`默认是离线分块诊断。`--stream PLAYBACK_ID`将原生具名绑定和原始权重通过标准输出传递给Python适配器；不写逐帧动画文件。单实例绑定独立播放ID，消费者必须校验绑定、时间戳、权重及版本。诊断通过不等于真机P0验收。
