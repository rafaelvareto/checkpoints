"""
MTCNN Detector
Based on: https://github.com/TropComplique/mtcnn-pytorch/
"""

import numpy
import torch

from PIL import Image
from torch.autograd import Variable
from typing import Tuple, List

from . import box_utils
from . import gpu
from .first_stage import run_first_stage
from .mtcnn_models import PNet, RNet, ONet


__all__ = ['MTCNN']


#pylint: disable=len-as-condition

class MTCNN(object):
    def __init__(
        self, 
        min_face_size: float = 20.0,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.8),
        nms_thresholds: Tuple[float, float, float] = (0.7, 0.7, 0.7),
        device: str = 'cpu') -> None:

        self._min_face_size = min_face_size
        self._thresholds = thresholds
        self._nms_thresholds = nms_thresholds

        self._pnet = PNet()
        self._rnet = RNet()
        self._onet = ONet()

        self._device = device
        self._pnet.to(self._device)
        self._rnet.to(self._device)
        self._onet.to(self._device)

        self._onet.eval()


    def __call__(self, image: numpy.ndarray) -> List:
        return self.detect(image)


    def detect(self, image: numpy.ndarray) -> List:
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.
        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # Skip conversion if image is an instance of PIL.Image
        if isinstance(image, numpy.ndarray):
            image = Image.fromarray(image)


        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that minimum size that we can detect equals to minimum face size that we want to detect
        min_length *= (min_detection_size/self._min_face_size)

        factor_count = 0
        while min_length > min_detection_size:
            scales.append((min_detection_size/self._min_face_size)*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # ========================= STAGE 1 =========================
        all_bboxes = []

        # run P-Net on different scales
        for scale in scales:
            boxes = run_first_stage(image, self._pnet, self._device,
                                    scale=scale, threshold=self._thresholds[0])
            all_bboxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        all_bboxes = [i for i in all_bboxes if i is not None]
        if len(all_bboxes) == 0:
            return []

        # it will be returned
        bounding_boxes: numpy.ndarray = numpy.vstack(all_bboxes)

        keep = box_utils.nms(bounding_boxes[:, 0:5], self._nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = box_utils.calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = box_utils.convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = numpy.round(bounding_boxes[:, 0:4])

        # ========================= STAGE 2 =========================
        img_boxes = box_utils.get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes))

        img_boxes = img_boxes.to(self._device)
        output = self._rnet(img_boxes)

        offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
        probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = numpy.where(probs[:, 1] > self._thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = box_utils.nms(bounding_boxes, self._nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = box_utils.calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = box_utils.convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = numpy.round(bounding_boxes[:, 0:4])

        # ========================= STAGE 3 =========================
        img_boxes = box_utils.get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return []
        img_boxes = Variable(torch.FloatTensor(img_boxes))

        img_boxes = img_boxes.to(self._device)
        output = self._onet(img_boxes)

        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
        probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = numpy.where(probs[:, 1] > self._thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # ======================= FINAL STAGE =======================
        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = numpy.expand_dims(xmin, 1) + numpy.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = numpy.expand_dims(ymin, 1) + numpy.expand_dims(height, 1)*landmarks[:, 5:10]

        bounding_boxes = box_utils.calibrate_box(bounding_boxes, offsets)
        keep = box_utils.nms(bounding_boxes, self._nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        detections = list()
        for bbox, keypoints in zip(bounding_boxes, landmarks):
            detections.append({
                'box': [int(bbox[0]), int(bbox[1]),
                        int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])],
                'confidence': bbox[-1],
                'keypoints': {
                    'left_eye': (int(keypoints[0]), int(keypoints[5])),
                    'right_eye': (int(keypoints[1]), int(keypoints[6])),
                    'nose': (int(keypoints[2]), int(keypoints[7])),
                    'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                    'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                }
            })

        return detections
