import torch
import torch.onnx

from data.config import set_cfg
from data.config import set_dataset

from yolact import Yolact

# source: https://michhar.github.io/convert-pytorch-onnx/

with torch.no_grad():
  # set darknet53 config and dataset
  set_dataset("arvp_general_dataset")
  #set_cfg("yolact_darknet53_config") # BROKE
  set_cfg("yolact_darknet53_arvp_general_config")

  # create model and set for inference
  model = Yolact()
  model.eval()

  # load weights
  state_dict = torch.load("./weights/yolact_darknet53_1114_68000_general.pth")

  # load weights into model
  model.load_state_dict(state_dict)

  # dummy input for forward pass
  dummy_input = torch.randn(1, 3, 550, 550)

  # export to ONNX
  # user Verbose=True for string output of graph
  # set opset_version to avoid conversion warning
  torch.onnx.export(model, dummy_input, "yolact_darknet53.onnx", opset_version=11)