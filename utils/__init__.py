from .objective import *
from .modifier import *
from .vocab import *
from .nsgaii import *
from .metric import *

torch.set_num_threads(1)
rdBase.DisableLog('rdApp.error')
dev = torch.device('cuda')
devices = [0]