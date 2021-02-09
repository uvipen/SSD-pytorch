"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import numpy as np
import argparse
import torch
from src.transform import SSDTransformer
import cv2
from PIL import Image

from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--input", type=str, required=True, help="the path to input video")
    parser.add_argument("--cls-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default=None, help="the path to output video")
    args = parser.parse_args()
    return args


def test(opt):
    model = SSD(backbone=ResNet())
    checkpoint = torch.load(opt.pretrained_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    dboxes = generate_dboxes()
    transformer = SSDTransformer(dboxes, (300, 300), val=True)
    cap = cv2.VideoCapture(opt.input)
    if opt.output is None:
        output = "{}_prediction.mp4".format(opt.input[:-4])
    else:
        output = opt.output
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    encoder = Encoder(dboxes)
    while cap.isOpened():
        flag, frame = cap.read()
        output_frame = np.copy(frame)
        if flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break
        frame = Image.fromarray(frame)
        frame, _, _, _ = transformer(frame, None, torch.zeros(1, 4), torch.zeros(1))

        if torch.cuda.is_available():
            frame = frame.cuda()
        with torch.no_grad():
            ploc, plabel = model(frame.unsqueeze(dim=0))
            result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > opt.cls_threshold).squeeze(axis=1)
            loc = loc[best]
            label = label[best]
            prob = prob[best]
            if len(loc) > 0:
                loc[:, 0::2] *= width
                loc[:, 1::2] *= height
                loc = loc.astype(np.int32)
                for box, lb, pr in zip(loc, label, prob):
                    category = coco_classes[lb]
                    color = colors[lb]
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)
                    text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(output_frame, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                                  -1)
                    cv2.putText(
                        output_frame, category + " : %.2f" % pr,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)

        out.write(output_frame)
    cap.release()
    out.release()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
