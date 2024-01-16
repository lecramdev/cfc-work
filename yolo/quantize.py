import argparse
from copy import deepcopy
from brevitas.quant_tensor import _unpack_quant_tensor
import torch
import torch.nn as nn
import onnx

import brevitas
from brevitas import config
import brevitas.graph
import brevitas.graph.utils
from brevitas.graph.quantize import quantize, preprocess_for_quantize, align_input_quant
from brevitas.graph.quantize_impl import are_inputs_unsigned, inp_placeholder_handler, layer_handler, add_output_quant_handler, layer_handler, recursive_input_handler, residual_handler
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from brevitas.export import export_qonnx
import brevitas.nn as qnn
import brevitas.quant as quant

from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.visualization import showInNetron

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

from PIL import Image

import inference.inference_util as inference_util

from tqdm import tqdm

class QuantizeWrapper(torch.nn.Module):
  def __init__(self, module: tasks.DetectionModel) -> None:
    super().__init__()
    self.m = module

  def forward(self, x):
    return self.m(x)
    
def insert_inp_quant(model, quant_identity_map):
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            # Insert UINT4 quantization at the input
            act_quant, kwargs_act_quant = quant_identity_map["unsigned"]
            inp_quant = act_quant(**kwargs_act_quant)
            name = node.name + '_quant'
            model.add_module(name, inp_quant)
            rewriters.append(brevitas.graph.InsertModuleCallAfter(name, node))
            # Insert UINT8 quantization at the input
            if "unsigned8" in quant_identity_map:
                act_quant, kwargs_act_quant = quant_identity_map["unsigned8"]
                inp_quant = act_quant(**kwargs_act_quant)
                name = node.name + '_8bit_quant'
                model.add_module(name, inp_quant)
                rewriters.append(brevitas.graph.InsertModuleCallAfter(name, node))
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model

def replace_outp_quant(model, quant_identity_map, quant_act_map, unsigned_act_tuple):
    rewriters = []
    for n in model.graph.nodes:
        if n.op == 'output':
            for node in n.all_input_nodes:
                if are_inputs_unsigned(model, node, [], quant_act_map, unsigned_act_tuple):
                    quant_module_class, quant_module_kwargs = quant_identity_map['unsigned8']
                else:
                    quant_module_class, quant_module_kwargs = quant_identity_map['signed8']
                quant_module = quant_module_class(**quant_module_kwargs)
                quant_module_name = node.name + '_8bit_quant'
                model.add_module(quant_module_name, quant_module)
                processed = [node.name]
                recursive_input_handler(
                    model,
                    node,
                    quant_module_name,
                    quant_module,
                    rewriters,
                    quant_identity_map,
                    align_input_quant,
                    align_sign=False,
                    path_list=[],
                    processed=processed)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model

def my_quantize(
        graph_model,
        quant_identity_map,
        compute_layer_map,
        quant_act_map,
        unsigned_act_tuple,
        requantize_layer_handler_output=True):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = graph_model.training
    graph_model.eval()
    graph_model = insert_inp_quant(graph_model, quant_identity_map)
    graph_model = layer_handler(graph_model, layer_map=quant_act_map, requantize_output=False)
    graph_model = add_output_quant_handler(graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple)
    graph_model = layer_handler(
        graph_model,
        layer_map=compute_layer_map,
        quant_identity_map=quant_identity_map,
        quant_act_map=quant_act_map,
        unsigned_act_tuple=unsigned_act_tuple,
        requantize_output=requantize_layer_handler_output)
    graph_model = residual_handler(graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple, align_input_quant)
    # graph_model = DisableLastReturnQuantTensor().apply(graph_model)
    graph_model = replace_outp_quant(graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple)
    graph_model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return graph_model

BIT_WIDTH = 6
compute_map = {
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': quant.Int8WeightPerTensorFloat,
            'weight_bit_width': BIT_WIDTH,
            # 'bias_quant': quant.Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': quant.Int8WeightPerTensorFloat,
            'weight_bit_width': BIT_WIDTH,
            # 'bias_quant': quant.Int32Bias,
            'return_quant_tensor': True}),
    nn.UpsamplingNearest2d: (
        qnn.QuantUpsamplingNearest2d,
        {})}
unsigned_act = (nn.ReLU,)
act_map = {
    nn.ReLU: (qnn.QuantReLU, {
        'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': BIT_WIDTH, 'return_quant_tensor': True})}
identity_map = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': quant.Int8ActPerTensorFloat, 'bit_width': BIT_WIDTH, 'return_quant_tensor': True}),
    'signed8':
        (qnn.QuantIdentity, {
            'act_quant': quant.Int8ActPerTensorFloat, 'bit_width': 8, 'return_quant_tensor': True}),
    'unsigned':
        (qnn.QuantIdentity, {
            'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': BIT_WIDTH, 'return_quant_tensor': True}),
    'unsigned8':
        (qnn.QuantIdentity, {
            'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': 8, 'return_quant_tensor': True})}

parser = argparse.ArgumentParser()
parser.add_argument("-o", nargs=1, default="models/quantized_yolo.onnx")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

yolo = YOLO("yolov6n.yaml", "detect")
model = torch.load("models/best.pt", map_location=torch.device('cpu'))["model"]
yolo.load(model)
model = yolo.model
model = model.float()
wrapper = QuantizeWrapper(model)
wrapper = wrapper.eval()
pre = preprocess_for_quantize(wrapper)
pre = brevitas.graph.ModuleToModuleByClass(nn.SiLU, nn.ReLU).apply(pre)
quantized = my_quantize(pre, identity_map, compute_map, act_map, unsigned_act, True)

SIZE = 256
dataloader = inference_util.get_dataloader("images", SIZE)
with torch.no_grad():
    print("Calibrate:")
    with calibration_mode(quantized):
        for x, _ in tqdm(dataloader):
            # quantized(torch.rand((1,3,SIZE,SIZE)))
            quantized(x)
    print("Bias Correction:")
    with bias_correction_mode(quantized):
        for x, _ in tqdm(dataloader):
            # quantized(torch.rand((1,3,SIZE,SIZE)))
            quantized(x)

export_qonnx(quantized, export_path=args.o[0], args=torch.rand((1, 3, SIZE, SIZE)))
# export_qonnx(quantized, export_path=args.o[0], args=torch.rand((1, 3, 8, 8)))
qonnx_cleanup(args.o[0], out_file=args.o[0])

if not args.test:
    exit()

class UnpackTensors(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        x = _unpack_quant_tensor(x)
        print(x[0][0,:,0,0])
        return x

my_detect = inference_util.QuantDetect(20, [8., 16., 32.])
inf_model = [
    quantized,
    UnpackTensors(),
    my_detect
]
for i, m in enumerate(inf_model):
    m.i = i
    m.f = -1
inf_model = torch.nn.Sequential(*inf_model)
inf_model.eval()
with torch.no_grad():
    for i, (x, _) in enumerate(dataloader):
        results = inference_util.infer(inf_model, x, SIZE)
        for res in results:
            annotated = res.plot()
            annotated = Image.fromarray(annotated)
            annotated.save(f"out_imgs/{i}.jpg")
        break

inference_util.print_stats()

yolo = YOLO("yolov6n.yaml", "detect")
yolo.model.model = inf_model
# yolo = YOLO("models/best.pt")
yolo.val(data="VOC.yaml", half=False, imgsz=SIZE, device="cpu")
