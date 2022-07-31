import PIL
from PIL import Image
import numpy as np
import torch
import os
import imageio


def pil_loader(path):
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


cx = 89
cy = 121

input_path = 'img_align_celeba/' 

output_path = 'img_celeba_cropped/'
if not os.path.isdir(output_path):
        os.mkdir(output_path)


img_list = [f for f in os.listdir(
        input_path) if os.path.splitext(f)[1] == ".jpg"]


for i, curr_path in enumerate(img_list):
    path = os.path.join(input_path, curr_path)
    img = np.array(pil_loader(path))
    img = img[cy - 64: cy + 64, cx - 64: cx + 64]
    save_path = os.path.join(output_path, curr_path)
    imageio.imwrite(save_path, img)
    if i % 100 == 0:
        print(i, '/', len(img_list))

