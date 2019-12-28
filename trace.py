import torch
import torch.onnx

import torch.backends.cudnn as cudnn

from data.config import set_cfg
from data.config import set_dataset

from yolact import Yolact

#from torch2trt import torch2trt

# source: https://michhar.github.io/convert-pytorch-onnx/

use_cuda = False #torch.cuda.is_available()

with torch.no_grad():
  if use_cuda:
    cudnn.fastest = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  # set darknet53 config and dataset
  set_dataset("arvp_general_dataset")
  #set_cfg("yolact_darknet53_config") # BROKE, IDK
  set_cfg("yolact_darknet53_arvp_general_config")

  # create model and set for inference
  model = Yolact()
  model.eval()

  # load weights
  state_dict = torch.load("./weights/yolact_darknet53_1114_68000_general.pth")
  model.load_state_dict(state_dict)
  
  if use_cuda:
    model = model.cuda()

  # dummy input for forward pass
  dummy_input = torch.randn(1, 3, 550, 550)

  if use_cuda:
    dummy_input = dummy_input.cuda()

  # export to ONNX
  # user Verbose=True for string output of graph
  # set opset_version to avoid conversion warning
  torch.onnx.export(model, dummy_input, "yolact_darknet53.onnx", opset_version=11)

  # export to TensorRT
  #model_trt = torch2trt(model, [dummy_input])