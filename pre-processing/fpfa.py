"""
Five-Points Face Aligner: This class performs face alignmet by five reference points
Based on:  https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
Insights:  https://stackoverflow.com/questions/25895587/python-skimage-transform-affinetransform-rotation-center
Reference: https://github.com/grib0ed0v/face_recognition.pytorch/blob/develop/utils/face_align.py
"""

import numpy
import skimage
import warnings

from skimage import transform
from typing import Tuple


class FivePointsFaceAligner:
    __ref_keypoints = numpy.array([
        30.2946 / 96, 51.6963 / 112,
        65.5318 / 96, 51.5014 / 112,
        48.0252 / 96, 71.7366 / 112,
        33.5493 / 96, 92.3655 / 112,
        62.7299 / 96, 92.2041 / 112], dtype=numpy.float64).reshape(5, 2)

    def __init__(
        self, 
        ref_keypoints: numpy.ndarray = __ref_keypoints,
        output_size: Tuple[int, int] = (400, 400), 
        padding: int = 0) -> None:

        self._ref_keypoints = ref_keypoints
        self._output_size = output_size
        self._padding = padding


    def __call__(self, image: numpy.ndarray, keypoints: numpy.ndarray):
        return self.align(image, keypoints)


    def align(self, image: numpy.ndarray, keypoints: numpy.ndarray):
        if not isinstance(image, numpy.ndarray):
            image = numpy.array(image)

        keypoints = numpy.array(keypoints).reshape(5, 2)
        output_height, output_width = self._output_size
        output_height = output_height - 2 * self._padding
        output_width = output_width - 2 * self._padding

        keypoints = keypoints.copy().astype(numpy.float64)

        keypoints_ref = numpy.zeros((5, 2), dtype=numpy.float64)
        keypoints_ref[:, 0] = (self._ref_keypoints[:, 0] * output_width) + self._padding
        keypoints_ref[:, 1] = (self._ref_keypoints[:, 1] * output_height) + self._padding

        transform_matrix = transform.estimate_transform('affine', keypoints_ref, keypoints)
        output_image = transform.warp(
            image, transform_matrix,
            output_shape=(self._output_size[0], self._output_size[0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_image = skimage.img_as_ubyte(output_image)
        return output_image, transform_matrix
        