import numpy
import torchvision

from .fpfa import FivePointsFaceAligner as FPFA
from .mtcnn.mtcnn import MTCNN
from ..networks import architectures

__all__ = ['default_transform', 'ToNumpy', 'FaceDetectAlign']


def default_transform(arch_name='oxford_resnet50'):
    """
    Provides default image transforms for the architectures given below, such as input image size and channel normalization.\n
        - AFFFE:    Eclipse - Ensembles of Centroids Leveraging Iteratively Processed Spatial Eclipse Clustering
        - ARCFACE:  Additive Angular Margin Loss for Deep Face Recognition
        - IMAGENET: ImageNet - A large-scale hierarchical image database
        - VGGFACE2: VGGFace2 - A dataset for recognising faces across pose and age
    """

    # WARNING:  Define employed feature transforms [Each architecture requires a different input size as well as specific means and stdevs]
    #           - For AFFFE, faces are not tightly cropped and, therefore, face detector is not employed by default.
    transform_dict = {
        'afffe': torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            ToNumpy(return_type='float'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.000] * 3, 
                 std=[255.0] * 3
            )
        ]),
        'arcface': torchvision.transforms.Compose([
            FaceDetectAlign(size=(112,112), return_type='float', scale_ratio=1.5, device='cpu'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[127.5] * 3, 
                 std=[128.0] * 3
            )
        ]),
        'imagenet': torchvision.transforms.Compose([
            FaceDetectAlign(size=(224,224), return_type='int', scale_ratio=1.5, device='cpu'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]
            )
        ]),
        'vggface2': torchvision.transforms.Compose([
            FaceDetectAlign(size=(224,224), return_type='float', scale_ratio=1.5, device='cpu'),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[131.0912, 103.8827, 91.4953], 
                 std=[1.0] * 3
            )
        ]),
    }
    if arch_name in ['arcface_iresnet34', 'arcface_iresnet50', 'arcface_iresnet100']:
        return transform_dict['arcface'] 
    elif   arch_name in ['cydonia_resnet50', 'cydonia_senet50', 'oxford_resnet50', 'oxford_senet50']:
        return transform_dict['vggface2'] 
    elif arch_name in ['vareto_mobilenet_v3_large', 'vareto_mobilenet_v3_small']:
        return transform_dict['imagenet']
    elif arch_name in ['vastlab_afffe']:
        return transform_dict['afffe']
    else: 
        raise ValueError(f'Invalid facesnet architecture type. Expected one of: {architectures.values()}')


class ToNumpy:
    """
    Convert a ``PIL Image`` to ``numpy.ndarray`` as uint8 or float32 type.\n
    It is used mainly to avoid unwanted ``torchvision.transforms.ToTensor`` rescaling integer images from range [0, 255] into ``torch.FloatTensor`` in the range [0.0, 1.0].\n
    Torch rescaling takes place when the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8.
    """
    def __init__(self, return_type='float'):
        assert (return_type=='float' or return_type=='int')
        self.return_type = numpy.float32 if return_type == 'float' else numpy.uint8

    def __call__(self, sample):
        return numpy.asarray(sample).astype(self.return_type)


class FaceDetectAlign:
    """
    Detect faces using ``MTCNN`` and performs a five-point face aligment.\n
    When faces are not found, it rescales the image by a given ratio and returns the image with a predefined size.\n
    It also runs the ``ToNumpy`` operation in order to convert a ``PIL Image`` to ``numpy.ndarray`` as uint8 or float32 type.
    """
    def __init__(self, size=(224,224), return_type='float', scale_ratio=1.5, device='cpu'):
        assert isinstance(size, (int,tuple)) and (return_type=='float' or return_type=='int')
        self.output_size = size
        self.return_type = numpy.float32 if return_type == 'float' else numpy.uint8
        self.scale_ratio = scale_ratio

        self.detector    = MTCNN(device=device)
        self.aligner     = FPFA(output_size=size)

    def __call__(self, sample):
        """performing image operations: handling grayscale images + fiducial point detection and alignment"""
        def align_image(image, aligner, bbox, keypoints):
            temp_keypoints = [
                keypoints['left_eye'],
                keypoints['right_eye'],
                keypoints['nose'],
                keypoints['mouth_left'],
                keypoints['mouth_right'],
            ]    
            image, rotation_matrix = aligner(image, temp_keypoints)
            if bbox is not None:
                p_x, p_y, width, height = tuple(bbox)
                points = numpy.array([
                    [p_x, p_y], [p_x + width, p_y],
                    [p_x + width, p_y + height], [p_x, p_y + height]
                ])
                transformed_points = rotation_matrix.inverse(points)
                max_x, max_y = tuple(numpy.max(transformed_points, axis=0))
                min_x, min_y = tuple(numpy.min(transformed_points, axis=0))
                bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
            if keypoints is not None:
                names, points = zip(*keypoints.items())
                points = numpy.asarray(points)
                transformed_points = rotation_matrix.inverse(points)
                transformed_points = map(tuple, transformed_points)
                keypoints = dict(zip(names, list(transformed_points)))
            return {'image':image, 'bbox':bbox, 'keys':keypoints}
        
        detection = self.detector.detect(sample)
        # If a single face is detected at least, proceed with alignment
        if len(detection):
            aligment = align_image(sample, self.aligner, detection[0]['box'], detection[0]['keypoints'])
            return aligment['image'].astype(self.return_type)
        else:
            if   isinstance(self.output_size, int):   new_size = int(self.output_size * self.scale_ratio)
            elif isinstance(self.output_size, tuple): new_size = (int(self.output_size[0] * self.scale_ratio), int(self.output_size[1] * self.scale_ratio))
            transform01 = torchvision.transforms.Resize(size=new_size)
            transform02 = torchvision.transforms.CenterCrop(size=self.output_size)
            transformed = transform02(transform01(sample))
            return numpy.asarray(transformed).astype(self.return_type)
