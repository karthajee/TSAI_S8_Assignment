import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):

  def __init__(self, in_C, out_C, k=3, 
               padding="same", p=0.1, 
               norm="bn", group_size=4, last=False):
    
    # Initialize super class
    super(ConvLayer, self).__init__()
    self.conv = nn.Conv2d(in_C, out_C, kernel_size=k, padding=padding, bias=False)
    if norm == "bn":
      self.norm = nn.BatchNorm2d(out_C)
    elif norm == 'gn':
      self.norm = nn.GroupNorm(group_size, out_C)
    elif norm == 'ln':
      self.norm = nn.GroupNorm(1, out_C)
    self.dout = nn.Dropout(p)
    self.last=last

  def __call__(self, x):

    x = self.conv(x)
    if not self.last:
      x = self.norm(x)
      x = self.dout(x)
      x = F.relu(x)
    return x

# Defining the model architecture
class Net(nn.Module):

  def __init__(self, b1_c=8, b2_c=16, b3_c=32, norm='bn', group_size=4):

    super().__init__()
    self.b1_C = b1_c
    self.b2_C = b2_c
    self.b3_C = b3_c

    # n_in: 32, r_in: 1, s: 1, j_in: 1 >> n_out: 32, r_out: 3, j_out: 1
    self.C1 = ConvLayer(3, self.b1_C)
    # n_in: 32, r_in: 3, s: 1, j_in: 1 >> n_out: 32, r_out: 5, j_out: 1
    self.C2 = ConvLayer(self.b1_C, self.b1_C)
    # n_in: 32, r_in: 5, s: 1, j_in: 1 >> n_out: 32, r_out: 5, j_out: 1
    self.c3 = ConvLayer(self.b1_C, self.b1_C, 1, 0)
    # n_in: 32, r_in: 5, s: 2, j_in: 1 >> n_out: 16, r_out: 6, j_out: 2
    self.P1 = nn.MaxPool2d(2, 2)

    
    # n_in: 16, r_in: 6, s: 1, j_in: 2 >> n_out: 16, r_out: 10, j_out: 2
    self.C4 = ConvLayer(self.b1_C, self.b2_C)
    # n_in: 16, r_in: 10, s: 1, j_in: 2 >> n_out: 16, r_out: 14, j_out: 2
    self.C5 = ConvLayer(self.b2_C, self.b2_C)
    # n_in: 16, r_in: 14, s: 1, j_in: 2 >> n_out: 16, r_out: 18, j_out: 2
    self.C6 = ConvLayer(self.b2_C, self.b2_C)
    # n_in: 16, r_in: 18, s: 1, j_in: 2 >> n_out: 16, r_out: 18, j_out: 2
    self.c7 = ConvLayer(self.b2_C, self.b2_C, 1, 0)
    # n_in: 16, r_in: 18, s: 2, j_in: 2 >> n_out: 8, r_out: 20, j_out: 4
    self.P2 = nn.MaxPool2d(2, 2)

    # n_in: 8, r_in: 20, s: 1, j_in: 4 >> n_out: 8, r_out: 28, j_out: 4
    self.C8 = ConvLayer(self.b2_C, self.b3_C)
    # n_in: 8, r_in: 28, s: 1, j_in: 4 >> n_out: 8, r_out: 36, j_out: 4
    self.C9 = ConvLayer(self.b3_C, self.b3_C)
    # n_in: 8, r_in: 36, s: 1, j_in: 4 >> n_out: 8, r_out: 44, j_out: 4
    self.C10 = ConvLayer(self.b3_C, self.b3_C)
    self.gap = nn.AdaptiveAvgPool2d((1, 1))
    # n_in: 1, r_in: 20, s: 1, j_in: 4 >> n_out: 8, r_out: 28, j_out: 4
    self.c11 = ConvLayer(self.b3_C, 10, 1, 0, last=True)

  def forward(self, x):
    
    x = self.C1(x)
    x = x + self.C2(x)
    x = x + self.c3(x)
    x = self.P1(x)

    x = self.C4(x)
    x = x + self.C5(x)
    x = x + self.C6(x)
    x = x + self.c7(x)
    x = self.P2(x)

    x = self.C8(x)
    x = x + self.C9(x)
    x = x + self.C10(x)
    x = self.gap(x)
    x = self.c11(x).squeeze()
    return F.log_softmax(x)