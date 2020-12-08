import sys
import collections
import math
import numbers
import random
import warnings

from PIL import Image

import torch
import torchvision.transforms.functional as F

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, writer=None):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.writer = writer

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop if 10 attempts are failed
        in_ratio = float(img.size[0]) / float(img.size[1])
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2 # the coordinate of top-left
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, images):
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)
        return [F.resized_crop(image, i, j, h, w, self.size, self.interpolation) for image in images]


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5, writer=None):
        self.p = p
        self.writer = writer

    def __call__(self, images):
        if random.random() < self.p:
            return [F.hflip(image) for image in images]
        if self.writer:
            self.writer.add_images('before hflip', images)
            self.writer.add_images('after hflip', [F.hflip(image) for image in images])
        return images

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, images):
        return [F.resize(image, self.size, self.interpolation) for image in images]

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, images):
        return [F.center_crop(image, self.size) for image in images]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class ToTensor(object):
    def __call__(self, images):
        tensors = [F.to_tensor(image) for image in images]
        return torch.stack(tensors, dim=1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def normalize(tensor, mean, std, inplace=False, writer=None):
    # if not inplace:
    #     tensor = tensor.clone()
    # print(tensor[:,0,:,:])
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None, None])
    tensor.div_(std[:, None, None, None])
    return tensor
    
class Normalize(object):
    def __init__(self, mean, std, inplace=False, writer=None):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.writer = writer

    def __call__(self, tensor):
        return normalize(tensor, self.mean, self.std, self.inplace, self.writer)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    com = Compose([ToTensor()])
    print(com)