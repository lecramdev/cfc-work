import time
import PIL
import torch
import numpy as np
from ultralytics import YOLO

import inference_util
import driver_base
import driver

class YoloFPGA(torch.nn.Module):
    def __init__(self, bitfile: str, platform: str = "alveo", device = None, batch_size: int = 1) -> None:
        super().__init__()
        self.accel = driver_base.FINNExampleOverlay(
            bitfile_name=bitfile,
            platform=platform,
            io_shape_dict=driver.io_shape_dict,
            batch_size=batch_size,
            runtime_weight_dir="runtime_weights/",
            device=device)

    def forward(self, x: torch.Tensor):
        start = time.time()
        dev = x.device
        x = torch.permute(x, (0, 2, 3, 1))
        x = x * 255
        x = x.cpu().numpy().astype(np.uint8)
        ibuf_normal = [x]
        obuf_normal = self.accel.execute(ibuf_normal)
        if not isinstance(obuf_normal, list):
            obuf_normal = [obuf_normal]
        y = [torch.from_numpy(buf).to(dev) for buf in obuf_normal]
        coeffs = [0.13880908489227295, 0.19925671815872192, 0.18177929520606995]
        y = [yi.permute(0, 3, 1, 2) * c for yi, c in zip(y, coeffs)]
        end = time.time()
        print(f"Time: {(end-start) * 1000}ms")
        return y

def create_inference_model(bitfile: str):
    fpga = YoloFPGA(bitfile)
    fpga.i = 0
    fpga.f = -1
    detect = inference_util.QuantDetect()
    detect.i = 1
    detect.f = -1
    return torch.nn.Sequential(fpga, detect)

SIZE = 256
dataloader = inference_util.get_dataloader("../../../images", SIZE)
inf_model = create_inference_model("../bitfile/finn-accel.xclbin")
inf_model.eval()
with torch.no_grad():
    for i, (x, _) in enumerate(dataloader):
        start = time.time()
        results = inference_util.infer(inf_model, x, SIZE)
        end = time.time()
        print(f"Total time: {(end-start) * 1000}ms")
        for res in results:
            annotated = res.plot()
            annotated = PIL.Image.fromarray(annotated)
            annotated.save(f"../../../out_imgs_fpga/{i}.jpg")
        # break

inference_util.print_stats()

# yolo = YOLO("yolov6n.yaml", "detect")
# yolo.model.model = inf_model
# # yolo = YOLO("models/best.pt")
# yolo.val(data="VOC.yaml", half=False, imgsz=SIZE, device="cpu")