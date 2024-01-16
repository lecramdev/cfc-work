import torch
import ultralytics
import ultralytics.nn
from ultralytics.utils import ops
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from torchvision import datasets, transforms

NAMES = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tvmonitor"
}

def get_dataloader(folder, size):
    # dataset = datasets.ImageFolder(folder, transforms.ToTensor())
    dataset = datasets.ImageFolder(folder, transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ]))
    return torch.utils.data.DataLoader(dataset)

global_mean = []
global_count = []

def print_stats():
    mean = 0
    count = 0
    for m, c in zip(global_mean, global_count):
        mean += m * c
        count += c
    if count == 0:
        print("No Detections!")
    else:
        print(f"Mean: {mean/count}, Count: {count}")

def postprocess(preds, img, orig_imgs, names):
    """Post-processes predictions and returns a list of Results objects."""
    preds = ops.non_max_suppression(preds, conf_thres=0.3, max_det=300)

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for i, pred in enumerate(preds):
        # print(pred.shape)
        if pred.shape[0] > 0:
            global_mean.append(pred[:,4].mean().item())
            global_count.append(pred.shape[0])
        orig_img = orig_imgs[i]
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path="", names=names, boxes=pred))
    return results

def infer(model, x, size):
    # transform = transforms.Compose([
    #     transforms.Resize(size),
    #     transforms.CenterCrop(size)
    # ])
    # t = transform(x)
    t = x
    y = model(t)
    if isinstance(y, tuple):
        y = y[0]
    return postprocess(y, t, t, NAMES)


class QuantDetect(torch.nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=20, stride=[ 8., 16., 32.]):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.FloatTensor(stride)  # strides computed during build
        self.dfl = ultralytics.nn.modules.block.DFL(self.reg_max) if self.reg_max > 1 else torch.nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.training:
            return x
        shape = x[0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, x

