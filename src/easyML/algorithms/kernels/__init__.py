from .one_vs_rest import OVR
from .multinomial import Multinomial

KERNELS = {'OVR': OVR(),\
            'MULTINOMIAL': Multinomial()}