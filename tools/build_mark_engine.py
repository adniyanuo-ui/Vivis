"""Build an RTX 4090 TensorRT engine from the official Mark ONNX model."""
from pathlib import Path
import tensorrt as trt
root=Path(__file__).resolve().parent.parent
folder=root/'assets/models/mark-v2.3'
logger=trt.Logger(trt.Logger.INFO)
with trt.Builder(logger) as builder, builder.create_network(0) as network, trt.OnnxParser(network,logger) as parser:
    if not parser.parse_from_file(str(folder/'network.onnx')):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    profile=builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t=network.get_input(i)
        shape=tuple(t.shape)
        print('Input',t.name,shape,flush=True)
        minimum=tuple(1 if d<0 else d for d in shape)
        optimum=tuple(8 if d<0 else d for d in shape)
        maximum=tuple(128 if d<0 else d for d in shape)
        profile.set_shape(t.name,minimum,optimum,maximum)
    config.add_optimization_profile(profile)
    engine=builder.build_serialized_network(network,config)
    if engine is None:raise RuntimeError('TensorRT engine build failed')
    (folder/'network.trt').write_bytes(bytes(engine))
print('TensorRT engine ready',flush=True)
