import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def enable_oled():
    """Turn the OLED stats page off, then on. The server is on this Jetson."""
    import requests
    base_url = "http://127.0.0.1:8000/stats"
    for action in ("off", "on"):
        response = requests.get("%s/%s" % (base_url, action), timeout=5)
        print("%s:%s" % (action.upper(), response.status_code))