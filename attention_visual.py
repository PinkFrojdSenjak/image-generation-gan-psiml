import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import DCSAGenerator
from utils import denorm

torch.manual_seed(125)
fixed_z = torch.randn(1, 512)

model = DCSAGenerator()

model.load_state_dict(torch.load('./models/dcsagan - 2/390000_G.pth', map_location='cpu'))

model.eval()
with torch.no_grad():
    fake_image, attn = model(fixed_z)
    # plot the fake image
    #plt.imshow(denorm(fake_image[0]).permute(1, 2, 0).cpu().numpy())
    x = 40 #21
    y = 27 #26
    idx = y * 64 + x
    fig, ax = plt.subplots(1,2, figsize=(8, 8))
    ax[0].imshow(denorm(fake_image[0]).permute(1, 2, 0).cpu().numpy())
    # heatmap for intensity
    im = attn[0, idx, :].cpu().view(64,64).numpy()
    im = im - np.min(im)
    im = im / (np.max(im) - np.min(im))
    im = im * 255
    ax[1].imshow(im, cmap='gray', vmin=0, vmax=255)
    # plot a dot at the center of the heatmap
    plt.plot(x, y, '.', color='red', markersize=10)
    plt.show()

    fig.savefig('../gan slike/dcsagan_2.png', dpi = 300)