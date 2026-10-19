"""Build an RTX 4090 TensorRT engine from the official A2F ONNX model."""
from pathlib import Path
import tensorrt as trt
root=Path(__file__).resolve().parent.parent
import sys,re,json
folder=Path(sys.argv[1]).resolve()
logger=trt.Logger(trt.Logger.INFO)
with trt.Builder(logger) as builder, builder.create_network(0) as network, trt.OnnxParser(network,logger) as parser:
    if not parser.parse_from_file(str(folder/'network.onnx')):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    profile=builder.create_optimization_profile()
    info=json.loads((folder/'trt_info.json').read_text())
    defaults=info['defaults']
    shapes={}
    for argument in info['trt_build_param']['batch']:
        argument=argument.format(**defaults)
        kind,value=argument.split('=',1)
        key={'--minShapes':'min','--optShapes':'opt','--maxShapes':'max'}[kind]
        for item in value.split(','):
            name,dims=item.split(':',1)
            shapes.setdefault(name,{})[key]=tuple(map(int,dims.split('x')))
    for i in range(network.num_inputs):
        tensor=network.get_input(i)
        bounds=shapes[tensor.name]
        print('Input',tensor.name,tuple(tensor.shape),bounds,flush=True)
        profile.set_shape(tensor.name,bounds['min'],bounds['opt'],bounds['max'])
    config.add_optimization_profile(profile)
    engine=builder.build_serialized_network(network,config)
    if engine is None:raise RuntimeError('TensorRT engine build failed')
    (folder/'network.trt').write_bytes(bytes(engine))
print('TensorRT engine ready',flush=True)
