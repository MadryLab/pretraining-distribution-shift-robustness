import torch as ch
import torchvision


class ResizeWrapper(ch.nn.Module):
    def __init__(self, model, size):
        super().__init__()
        self.resize = torchvision.transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        self.center_crop = torchvision.transforms.CenterCrop(size)
        self.model = model

    def forward(self, x):
        x = self.resize(x)
        x = self.center_crop(x)
        return self.model(x)