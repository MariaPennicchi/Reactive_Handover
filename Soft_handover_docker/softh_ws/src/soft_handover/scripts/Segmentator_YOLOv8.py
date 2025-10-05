import torch
from ultralytics import YOLO
import numpy as np
import cv2

class Segmentator():
    def __init__(self, conf_thres, weights):
        self.model = YOLO(weights)  # load an official model
        # print(torch.cuda.is_available())
        # self.model.to(0)
        self.line_thickness = 0.1
        self.conf_thres = conf_thres

    # accepts a BGR image
    def predict(self, im: np.ndarray):

        masks = None
        im0 = im.copy()
        results = self.model.predict(im, conf=self.conf_thres, stream=False, verbose=False, device=0)
        classes = list()

        # Process predictions
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            if result.masks is not None:
                masks = result.masks.data
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy().astype(np.uint8)

                if im0.shape[0] > 0 and im0.shape[1] > 0:
                    if masks.ndim == 3:
                        resized_masks = []
                        for mask in masks:
                            resized = cv2.resize(mask, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
                            resized_masks.append(resized)
                        masks = np.stack(resized_masks, axis=0)
                        masks = np.transpose(masks, (1, 2, 0))
                    else:
                        masks = cv2.resize(masks, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
            im0 = result.plot()

            for i in range(len(boxes.cls.cpu().numpy())):
                classes.append(boxes.cls.cpu().numpy()[i])

        return im0, masks, np.array(classes)
        # array with object classes is also returned