from .nsgaii import *
from .metric import *
from .objective import *
from .modifier import *
from .vocab import *
from .optim import *

torch.set_num_threads(1)
rdBase.DisableLog('rdApp.info')
dev = torch.device('cuda')
devices = [0]

