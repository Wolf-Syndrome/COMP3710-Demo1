import torch
import numpy as np
import matplotlib.pyplot as plt

def checkinfo():
    print("PyTorch Version:", torch.__version__) # Expected 2.0.1
    print("GPU Available:", torch.cuda.is_available()) # Expected 'cuda' or true
    print(torch.version.cuda)

#checkinfo()

# Config Torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

def part2():
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

    # Create pytorch tensors and send to gpu
    # Create complex tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)
    zs = torch.zeros_like(z).to(device)
    ns = torch.zeros_like(z).to(device)
    z = z.to(device)

    # Mandelbrot Set, check if diverge within n iterations
    for i in range(200):
        # Compete the next value of z using the formula z new = z^2 + c (c is the original point and z is previous)
        zs_ = zs*zs + z
        # Check divergence
        not_diverged = torch.abs(zs_) < 4.0
        # Update variables
        ns += not_diverged.type(torch.float)
        zs = zs_
    
    # Show inital image
    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()


def high_res():
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    print("Begin high res algorithm")
    x_centre = 0.1 
    y_centre = -1.33
    giv = 0.1
    Y, X = np.mgrid[x_centre-giv:x_centre+giv:0.00005, y_centre-giv:y_centre+giv:0.00005]

    # Create pytorch tensors and send to gpu
    # Create complex tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    z = torch.complex(x, y)
    print("Sending stuff to gpu")
    zs = torch.zeros_like(z).to(device)
    ns = torch.zeros_like(z).to(device)
    z = z.to(device)

    print("Begin calculations for high res")
    # Mandelbrot Set, check if diverge within n iterations
    for i in range(200):
        # Compete the next value of z using the formula z new = z^2 + c (c is the original point and z is previous)
        zs_ = zs*zs + z
        # Check divergence
        not_diverged = torch.abs(zs_) < 4.0
        # Update variables
        ns += not_diverged.type(torch.float)
        zs = zs_
    
    # Show inital image
    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()

def julia():
    # https://www.karlsims.com/julia.html
    # Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
    #Y, X = np.mgrid[0.5:2.5:0.005, -1:1:0.005]
    Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]

    # Create pytorch tensors and send to gpu
    # Create complex tensors
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    zs = torch.complex(x, y).to(device)
    ns = torch.zeros_like(zs).to(device)
    real = 0.274
    imag = -0.008
    realA = torch.full_like(zs, real, dtype=torch.float)
    imagA = torch.full_like(zs, imag, dtype=torch.float)
    c = torch.complex(realA, imagA).to(device)


    # Mandelbrot Set, check if diverge within n iterations
    for i in range(200):
        # Compete the next value of z using the formula z new = z^2 + c (c is chosen and zs is previous)
        zs_ = zs*zs + c
        # Check divergence, this is 2 for julia set
        not_diverged = torch.abs(zs_) < 2.0
        # Update variables
        ns += not_diverged.type(torch.float)
        zs = zs_
    
    # Show inital image
    plt.imshow(processFractal(ns.cpu().numpy()))
    plt.tight_layout(pad=0)
    plt.show()

#part2()
#high_res()
julia()

