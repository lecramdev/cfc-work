import argparse
from copy import deepcopy
from brevitas.quant_tensor import _unpack_quant_tensor
import torch
import torch.nn as nn
# import onnx

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

# from qonnx.util.cleanup import cleanup as qonnx_cleanup
# from qonnx.core.modelwrapper import ModelWrapper
# from qonnx.core.datatype import DataType
# from qonnx.transformation.base import Transformation
# from qonnx.transformation.infer_shapes import InferShapes
# from qonnx.util.basic import get_by_name

# from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
# from finn.util.visualization import showInNetron

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
import ultralytics.nn.tasks as tasks

import warnings
warnings.filterwarnings("ignore", r"Defining your .*", category=UserWarning) 

from PIL import Image
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

import inference.inference_util as inference_util

from tqdm import tqdm

class QuantizeWrapper(torch.nn.Module):
  def __init__(self, module: tasks.DetectionModel) -> None:
    super().__init__()
    self.m = module

  def forward(self, x):
    return self.m(x)

class DetectWrapperQuantize(torch.nn.Module):
  def __init__(self, module) -> None:
    super().__init__()
    self.m = module

  def forward(self, x1, x2, x3):
    return self.m([x1, x2, x3])
  
class DetectWrapper(torch.nn.Module):
  def __init__(self, module) -> None:
    super().__init__()
    self.m = module

  def forward(self, x):
    return self.m(x[0], x[1], x[2])
    
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
        insert_input_quant,
        replace_output_quant,
        requantize_layer_handler_output=True):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = graph_model.training
    graph_model.eval()
    if insert_input_quant:
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
    if replace_output_quant:
        graph_model = replace_outp_quant(graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple)
    graph_model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return graph_model

ACT_BIT_WIDTH = 5
WEIGHT_BIT_WIDTH = 6
compute_map = {
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': quant.Int8WeightPerTensorFloat,
            'weight_bit_width': WEIGHT_BIT_WIDTH,
            # 'bias_quant': quant.Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': quant.Int8WeightPerTensorFloat,
            'weight_bit_width': WEIGHT_BIT_WIDTH,
            # 'bias_quant': quant.Int32Bias,
            'return_quant_tensor': True}),
    nn.UpsamplingNearest2d: (
        qnn.QuantUpsamplingNearest2d,
        {})}
unsigned_act = (nn.ReLU,)
act_map = {
    nn.ReLU: (qnn.QuantReLU, {
        'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': ACT_BIT_WIDTH, 'return_quant_tensor': True})}
identity_map = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': quant.Int8ActPerTensorFloat, 'bit_width': ACT_BIT_WIDTH, 'return_quant_tensor': True}),
    'signed8':
        (qnn.QuantIdentity, {
            'act_quant': quant.Int8ActPerTensorFloat, 'bit_width': 8, 'return_quant_tensor': True}),
    'unsigned':
        (qnn.QuantIdentity, {
            'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': ACT_BIT_WIDTH, 'return_quant_tensor': True}),
    'unsigned8':
        (qnn.QuantIdentity, {
            'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': 8, 'return_quant_tensor': True})}


ACT_BIT_WIDTH_DETECT = 6
WEIGHT_BIT_WIDTH_DETECT = 8
compute_map_detect = {
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': quant.Int8WeightPerTensorFloat,
            'weight_bit_width': WEIGHT_BIT_WIDTH_DETECT,
            # 'bias_quant': quant.Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': quant.Int8WeightPerTensorFloat,
            'weight_bit_width': WEIGHT_BIT_WIDTH_DETECT,
            # 'bias_quant': quant.Int32Bias,
            'return_quant_tensor': True}),
    nn.UpsamplingNearest2d: (
        qnn.QuantUpsamplingNearest2d,
        {})}
unsigned_act_detect = (nn.ReLU,)
act_map_detect = {
    nn.ReLU: (qnn.QuantReLU, {
        'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': ACT_BIT_WIDTH_DETECT, 'return_quant_tensor': True})}
identity_map_detect = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': quant.Int8ActPerTensorFloat, 'bit_width': ACT_BIT_WIDTH_DETECT, 'return_quant_tensor': True}),
    'signed8':
        (qnn.QuantIdentity, {
            'act_quant': quant.Int8ActPerTensorFloat, 'bit_width': 8, 'return_quant_tensor': True}),
    'unsigned':
        (qnn.QuantIdentity, {
            'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': ACT_BIT_WIDTH_DETECT, 'return_quant_tensor': True}),
    'unsigned8':
        (qnn.QuantIdentity, {
            'act_quant': quant.Uint8ActPerTensorFloat, 'bit_width': 8, 'return_quant_tensor': True})}
parser = argparse.ArgumentParser()
parser.add_argument("-o", nargs=1, default="models/quantized_yolo.onnx")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# yolo = YOLO("myyolov6n.yaml", "detect")
# # model = torch.load("models/best.pt", map_location=device)["model"]
# model = torch.load("quant_yolo/d0.167_w0.25_c1024_relu_yolo/weights/best.pt", map_location=device)["model"]
# yolo.load(model)
# model = yolo.model
# model = model.float()
# wrapper = QuantizeWrapper(model)
# wrapper = wrapper.eval()
# pre = preprocess_for_quantize(wrapper)
# # pre = brevitas.graph.ModuleToModuleByClass(nn.SiLU, nn.ReLU).apply(pre)
# quantized = pre #my_quantize(pre, identity_map, compute_map, act_map, unsigned_act, True)

SIZE = 320
# dataloader = inference_util.get_dataloader("images", SIZE)
# quantized.to(device)
# with torch.no_grad():
#     print("Calibrate:")
#     with calibration_mode(quantized):
#         for x, _ in tqdm(dataloader):
#             # quantized(torch.rand((1,3,SIZE,SIZE)))
#             x = x.to(device)
#             quantized(x)
#     print("Bias Correction:")
#     with bias_correction_mode(quantized):
#         for x, _ in tqdm(dataloader):
#             # quantized(torch.rand((1,3,SIZE,SIZE)))
#             x = x.to(device)
#             quantized(x)

# export_qonnx(quantized, export_path=args.o[0], args=torch.rand((1, 3, SIZE, SIZE)))
# # export_qonnx(quantized, export_path=args.o[0], args=torch.rand((1, 3, 8, 8)))
# qonnx_cleanup(args.o[0], out_file=args.o[0])

if not args.test:
    exit()

class UnpackTensors(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        x = _unpack_quant_tensor(x)
        # print(x[0][0,:,0,0])
        return x

# my_detect = inference_util.QuantDetect(20, [8., 16., 32.])
# inf_model = [
#     quantized,
#     UnpackTensors(),
#     my_detect
# ]
# for i, m in enumerate(inf_model):
#     m.i = i
#     m.f = -1
# inf_model = torch.nn.Sequential(*inf_model)
# inf_model.eval().to(device)
# dataloader = inference_util.get_dataloader("images", SIZE)
# with torch.no_grad():
#     for i, (x, _) in enumerate(tqdm(dataloader, desc="Sample images")):
#         results = inference_util.infer(inf_model, x.to(device), SIZE)
#         for res in results:
#             annotated = res.plot()
#             annotated = Image.fromarray(annotated)
#             annotated.save(f"out_imgs/{i}.jpg")
#         # break

# inference_util.print_stats()


###############################################################################

class MyDetectionModel(tasks.DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)

    def calibrate(self):
        quantized = self.model[0]
        quantized_detect = self.model[1]
        dataloader = inference_util.get_dataloader("images", SIZE)
        quantized.to(device)
        quantized_detect.to(device)
        with torch.no_grad():
            print("Calibrate:")
            with calibration_mode(quantized.m), calibration_mode(quantized_detect.m):
                for x, _ in tqdm(dataloader):
                    x = quantized(x.to(device))
                    quantized_detect(x)
            print("Bias Correction:")
            with bias_correction_mode(quantized.m), bias_correction_mode(quantized_detect.m):
                for x, _ in tqdm(dataloader):
                    x = quantized(x.to(device))
                    quantized_detect(x)
            # print("Calibrate Detect:")
            # with calibration_mode(quantized_detect):
            #     for x, _ in tqdm(dataloader):
            #         quantized(x.to(device))
            # print("Bias Correction Detect:")
            # with bias_correction_mode(quantized_detect):
            #     for x, _ in tqdm(dataloader):
            #         quantized(x.to(device))

    def quantize(self):
        detect = self.model[-1]
        self.model[-1] = torch.nn.Identity()
        self.model[-1].i = detect.i
        self.model[-1].f = detect.f

        wrapper = QuantizeWrapper(self)
        pre = preprocess_for_quantize(wrapper)
        # pre = brevitas.graph.ModuleToModuleByClass(nn.SiLU, nn.ReLU).apply(pre)
        quantized = my_quantize(pre, identity_map, compute_map, act_map, unsigned_act, True, False, True)

        wrapper_detect = DetectWrapperQuantize(detect)
        pre_detect = preprocess_for_quantize(wrapper_detect)
        # pre = brevitas.graph.ModuleToModuleByClass(nn.SiLU, nn.ReLU).apply(pre)
        quantized_detect = my_quantize(pre_detect, identity_map_detect, compute_map_detect, act_map_detect, unsigned_act_detect, False, True, True)

        # Replace Detect layer with custom Detect
        # detect = self.model[-1]
        quant_detect = inference_util.QuantDetect(nc=detect.nc, stride=detect.stride)
        # Rebuild Model
        self.model = torch.nn.Sequential(QuantizeWrapper(quantized), DetectWrapper(quantized_detect), UnpackTensors(), quant_detect)
        for i, m in enumerate(self.model):
            m.i = i
            m.f = -1
        self.save = []
        self.calibrate()

    def _predict_once(self, x, profile=False, visualize=False):
        # if isinstance(x, torch.Tensor):
        #     fig, (img_ax, hist_ax) = plt.subplots(1, 2)
        #     img = x.numpy(force=True)[0,:,:,:]
        #     qimg = (img.transpose((1, 2, 0)) * 255).astype(int) & 0xf0
        #     img_ax.imshow(qimg)
        #     # img_ax.imshow(img.transpose((1, 2, 0)))
        #     hist_ax.hist(img.reshape(-1), 128)
        #     fig.set_figwidth(18)
        #     fig.set_figheight(6)
        #     plt.show()
        # if not isinstance(x, torch.fx.Proxy):
        #     x = x * 2 - 1
        return super()._predict_once(x, profile, visualize)

def get_model(cfg=None, weights=None, nc=20, verbose=True):
    model = MyDetectionModel(cfg, nc=nc, verbose=verbose)
    if weights:
        model.load(weights)
    else:
        # model.load_state_dict(torch.load("weights.pth"))
        model.load(torch.load("quant_yolo/d0.167_w0.25_c1024_relu_yolo/weights/best.pt", map_location=device))
    print("Quantize Model")
    model.quantize()
    return model

class MyTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # self.add_callback("on_train_epoch_end", self.calibrate)

    def calibrate(self, *args):
        self.model.calibrate()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        return get_model(cfg, weights, self.data['nc'], verbose and RANK == -1)
    
    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        # Save last and best
        chkpt = self.model.state_dict()
        torch.save(chkpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(chkpt, self.best)
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            torch.save(chkpt, self.wdir / f'epoch{self.epoch}.pt')

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.model.load_state_dict(torch.load(f))
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=self.model)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        self.ema.enabled = False
        self.validator(model=deepcopy(self.model))

# get_model("myyolov6n.yaml")

yolo = YOLO("myyolov6n.yaml", "detect")
# # yolo.model.model = inf_model
# # # yolo = YOLO("models/best.pt")
# # yolo.val(data="VOC.yaml", half=False, imgsz=SIZE, device=device)

yolo.train(
    data='VOC.yaml', imgsz=SIZE,
    epochs=1, patience=5,
    plots=True, device=0, cache=False, amp=False,
    half=False,
    pretrained=True, resume=True,
    lr0=0.0001,
    momentum=0.9,
    warmup_epochs=0,
    optimizer="SGD",
    trainer=MyTrainer,
    # data='coco128.yaml',
    # freeze=freeze,
    #fraction=0.1,
    project="qat_yolo",
    name="test")