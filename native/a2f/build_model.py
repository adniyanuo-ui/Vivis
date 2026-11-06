"""Build a single-stream Orin engine from verified original A2F ONNX bytes."""
import hashlib
import json
import sys
from pathlib import Path
import time

import tensorrt as trt

bundle = Path(sys.argv[1]).resolve()
source = bundle
root = bundle.parent
result = dict(scope='A2F ONNX parse and native single-stream TensorRT build; no servo I/O',
              tensorrt=trt.__version__, model_sha256=hashlib.sha256((source/'network.onnx').read_bytes()).hexdigest(),
              max_batch_size=1, completed=False)
begin = time.monotonic()
try:
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    with trt.Builder(logger) as builder, builder.create_network(0) as network, trt.OnnxParser(network, logger) as parser:
        if not parser.parse_from_file(str(source / 'network.onnx')):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        result['parse_seconds'] = time.monotonic() - begin
        info = json.loads((source / 'trt_info.json').read_text())
        bounds = {}
        for argument in info['trt_build_param']['batch']:
            key, values = argument.format(MAX_BATCH_SIZE=1, OPT_BATCH_SIZE=1).split('=', 1)
            for entry in values.split(','):
                name, dimensions = entry.split(':', 1)
                bounds.setdefault(name, {})[key] = tuple(map(int, dimensions.split('x')))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        config.builder_optimization_level = 2
        builder.max_threads = 2
        profile = builder.create_optimization_profile()
        result['inputs'] = []
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            sizes = bounds[tensor.name]
            # TensorRT10.16 returns None on success and raises on invalid shapes.
            profile.set_shape(tensor.name, sizes['--minShapes'], sizes['--optShapes'], sizes['--maxShapes'])
            result['inputs'].append(dict(name=tensor.name, shape=list(tensor.shape),
                                         dtype=str(tensor.dtype), profile=sizes))
        config.add_optimization_profile(profile)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TensorRT returned no engine')
        engine = bytes(serialized)
        destination = bundle / 'network.trt'
        temporary = destination.with_suffix('.trt.partial')
        temporary.write_bytes(engine)
        temporary.replace(destination)
        result.update(completed=True, engine_bytes=len(engine), engine_sha256=hashlib.sha256(engine).hexdigest())
except Exception as exc:
    result['error'] = str(exc)
finally:
    result['wall_seconds'] = time.monotonic() - begin
    (root / 'model_build.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if result['completed'] else 1)
