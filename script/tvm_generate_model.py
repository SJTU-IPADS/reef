import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime
import sys
import json

#####################################################
#
# This is an example of how to generate source code 
# and schedule json from tvm. 
#
#####################################################


if len(sys.argv) != 5:
    print("Usage: device_source_file_name raw_schedule_file graph_json_file param_file")
    exit(0)

source_file = open(sys.argv[1], "w")
raw_schedule_file = open(sys.argv[2], "w")
graph_json_file = open(sys.argv[3], "w")
param_file = open(sys.argv[4], "w+b")

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape
)
# mod, params = relay.testing.mobilenet.get_workload(
#     batch_size=batch_size, image_shape=image_shape
# )

opt_level = 3
target = tvm.target.rocm()

with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)

ctx = tvm.rocm()
module = graph_runtime.GraphModule(lib["default"](ctx))

data = np.ones(data_shape).astype("float32")
data = data * 10
module.set_input("data", data)

module.run()

source_file.write(lib.get_lib().imported_modules[0].get_source("hip"))
source_file.close()

graph_json_file.write(lib.get_json())
graph_json_file.close()

raw_schedule_file.write(module.module["get_schedule_json"]())
raw_schedule_file.close()


def dump_params(params, f):
    import array
    magic = bytes("TVM_MODEL_PARAMS\0", "ascii")
    f.write(magic)
    f.write(array.array('Q',[len(params)]).tobytes())
    for k in params.keys():
        param = array.array('f', params[k].asnumpy().flatten().tolist())
        f.write(bytes(k, "ascii"))
        f.write(bytes("\0", "ascii"))
        f.write(array.array('Q',[len(param)]).tobytes())
        f.write(param.tobytes())

dump_params(params, param_file)    
param_file.close()

out = module.get_output(0, tvm.nd.empty(out_shape)).asnumpy()
print(out.flatten()[0:10])