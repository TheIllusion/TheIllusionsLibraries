import torch
import numpy as np
from torch.autograd import Variable
x = np.array([[0.1, 0.9, 0.8, 0.91]])

x = torch.from_numpy(x)

x = Variable(x)

print torch.max(x, 1)[1]