import torch
from ultralytics import YOLO

def save_weights():
    saved = torch.load("quant_yolo/train_yolo/weights/best.pt")
    state_dict = saved.get("model").state_dict()
    torch.save(state_dict, "weights.pth")

# def validate_quant():
#     model = YOLO("yolov6n.yaml", "detect")
#     model.model = get_model("yolov6n.yaml", None, 20)
#     model.val(data='VOC.yaml', half=False, plots=True)

def validate_float():
    model = YOLO("quant_yolo/train_yolo/weights/best.pt", "detect")
    model.val(data='VOC.yaml', imgsz=256, half=False, plots=True)

def validate_float2():
    model = YOLO("myyolov6n.yaml", "detect")
    model.load("quant_yolo/train_yolo/weights/best.pt")
    model.val(data='VOC.yaml', half=False, plots=True)

def export_model():
    model = YOLO("myyolov6n.yaml", "detect")
    model.export(format="onnx", imgsz=256, simplify=True)

def train_model():
    model = YOLO("myyolov6n.yaml", "detect")
    model.model.info(detailed=False, verbose=True, imgsz=256)
    # model.load("quant_yolo/d0.167_w0.25_c1024_relu_yolo/weights/best.pt")
    try:
        # freeze = list(range(29))
        # freeze.remove(9)
        model.train(
            data='VOC.yaml', epochs=1000, imgsz=256,
            plots=True, device=0, cache=True, amp=False,
            pretrained=True, resume=True,
            lr0=0.001,
            optimizer="Adam",
            # trainer=MyTrainer,
            # data='coco128.yaml',
            # freeze=freeze,
            # fraction=0.1,
            project="quant_yolo",
            name="d0.167_w0.25_c1024_relu_scratch_yolo")
    except FileNotFoundError:
        pass

# export_model()
train_model()
# validate_float()
# validate_float2()
# save_weights()
# validate_quant()
