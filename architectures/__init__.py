# https://www.robots.ox.ac.uk/~albanie/pytorch-models.html
# https://www.github.com/cydonia999/VGGFace2-pytorch/
# https://www.github.com/nizhib/pytorch-insightface


from .cydonia_resnet50_ft_vggface2 import resnet50_ft_vggface2 as cydonia_resnet50
from .cydonia_senet50_ft_vggface2  import senet50_ft_vggface2  as cydonia_senet50

from .insight_iresnet_arcface import iresnet34_arcface  as insight_arcface34
from .insight_iresnet_arcface import iresnet50_arcface  as insight_arcface50
from .insight_iresnet_arcface import iresnet100_arcface as insight_arcface100

from .oxford_resnet50_ft_vggface2 import resnet50_ft_vggface2 as oxford_resnet50
from .oxford_senet50_ft_vggface2  import senet50_ft_vggface2  as oxford_senet50

from .vareto_mobilenetv3_ft_vggface2 import mobilenet_v3_large as vareto_mobilenet_v3_large
from .vareto_mobilenetv3_ft_vggface2 import mobilenet_v3_small as vareto_mobilenet_v3_small

from .vastlab_resnet51_affffe import resnet51_afffe as vastlab_afffe


architectures = [
    'cydonia_resnet50', 'cydonia_senet50', 
    'insight_arcface34', 'insight_arcface50', 'insight_arcface100',
    'oxford_resnet50', 'oxford_senet50',
    'vareto_mobilenet_v3_large', 'vareto_mobilenet_v3_small',
    'vastlab_afffe',
]
