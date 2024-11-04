#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np
import time

import tensorrt as trt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)

def main():
    # Build a TensorRT engine.
    # engine = build_engine_onnx("/home/qxy/dppo/trotting_export.onnx")

    # with open('./trotting.trt', 'wb') as f:
    #     f.write(bytearray(engine.serialize()))
    runtime = trt.Runtime(TRT_LOGGER)
    with open('/home/qxy/dppo/trotting.trt', 'rb') as f:
        engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Contexts are used to perform inference.
    context = engine.create_execution_context()

    inp = np.zeros((1,1170))
    inp[0,-39:] = np.array([-0.01817463,  0.00316524, -0.99982983, -0.001551,    0.00625625, -0.02259382,
  0.00605177,  0.03229075, -0.01595549, -0.00646622, -0.01030582, -0.03389338,
  0.00265833, -0.0066327,  -0.02148847, -0.00350176, -0.04536879,  0.08309361,
  0.00468482, -0.06002165,  0.10744572, -0.0034748,  -0.05103638,  0.08126458,
  0.00258044, -0.03308586,  0.05241467,  0.,          0.       ,   0.,
  0.    ,      0.   ,       0. ,         0.  ,        0.,          0.,
  0.   ,       0.    ,      0.        ]
)
    
    np.copyto(inputs[0].host, inp)

    t = []
    for i in range(10):
        t1 = time.time()
        trt_outputs = common.do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        print(time.time()-t1)
        t.append(time.time()-t1)
    # We use the highest probability as our prediction. Its index corresponds to the predicted label.
    print(sum(t)/len(t))
    print(trt_outputs[0][:24])
    common.free_buffers(inputs, outputs, stream)


if __name__ == "__main__":
    main()
