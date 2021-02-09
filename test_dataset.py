"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
import numpy as np
import argparse
import torch
from src.dataset import CocoDataset
from src.transform import SSDTransformer
import cv2
import shutil

from src.utils import generate_dboxes, Encoder, colors
from src.model import SSD, ResNet


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/coco", help="the root folder of dataset")
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default="predictions")
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
    test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, (300, 300), val=True))
    encoder = Encoder(dboxes)

    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    for img, img_id, img_size, _, _ in test_set:
        if img is None:
            continue
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            ploc, plabel = model(img.unsqueeze(dim=0))
            result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > opt.cls_threshold).squeeze(axis=1)
            loc = loc[best]
            label = label[best]
            prob = prob[best]
            if len(loc) > 0:
                path = test_set.coco.loadImgs(img_id)[0]["file_name"]
                output_img = cv2.imread(os.path.join(opt.data_path, "val2017", path))
                height, width, _ = output_img.shape
                loc[:, 0::2] *= width
                loc[:, 1::2] *= height
                loc = loc.astype(np.int32)
                for box, lb, pr in zip(loc, label, prob):
                    category = test_set.label_info[lb]
                    color = colors[lb]
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
                    text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                                  -1)
                    cv2.putText(
                        output_img, category + " : %.2f" % pr,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)
                    cv2.imwrite("{}/{}_prediction.jpg".format(opt.output, path[:-4]), output_img)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
