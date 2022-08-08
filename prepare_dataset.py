import PIL
from PIL import Image
import numpy as np
import torch
import os
import imageio
import cv2

def pil_loader(path):
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


cx = 100
cy = 130

input_path = 'psiml_data/' 

output_path = 'psiml_data_cropped/'
if not os.path.isdir(output_path):
        os.mkdir(output_path)


img_list = [f for f in os.listdir(
        input_path) if os.path.splitext(f)[1] == ".jpg"]


for i, curr_path in enumerate(img_list):
    path = os.path.join(input_path, curr_path)
    img = np.array(pil_loader(path))
    h, w, c = img.shape
    img = img[:min(h, w) - 1, :min(h, w) - 1, :]
    # resize to 256x256
    img = cv2.resize(img, (256, 256))
    img = img[cy - 64: cy + 64, cx - 64: cx + 64]
    save_path = os.path.join(output_path, curr_path)
    imageio.imwrite(save_path, img)
    if i % 20 == 0:
        print(i, '/', len(img_list))

