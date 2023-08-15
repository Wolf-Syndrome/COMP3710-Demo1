#Don't forget to add packages to conda `COMP3710-D1`

import torch
import numpy as np
import matplotlib.pyplot as plt

def checkinfo():
    print("PyTorch Version:", torch.__version__) # Expected 2.0.1
    print("GPU Available:", torch.cuda.is_available()) # Expected 'cuda' or true
#checkinfo()


# Config Torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Learning pytorch tensors, offloading to gpu
def part1():
    # Create coordinates
    X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

    #Convert numpy arrays to tensor arrays then send to GPU
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    x = x.to(device)
    y = y.to(device)

    z_gaus = torch.exp(-(x**2+y**2)/2.0)
    z_sine = torch.sin(x+y)
    z_gabor_filter = z_gaus.mul(z_sine)

    # Show plot
    if True:
        plt.imshow(z_gaus.cpu().numpy())
        plt.tight_layout()
        plt.show()

    # Show plot
    plt.imshow(z_sine.cpu().numpy())
    plt.tight_layout()
    plt.show()

    # Show plot
    plt.imshow(z_gabor_filter.cpu().numpy())
    plt.tight_layout()
    plt.show()
part1()