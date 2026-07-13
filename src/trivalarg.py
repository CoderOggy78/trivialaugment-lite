import random
from PIL import Image, ImageOps, ImageEnhance

PARAMETER_MAX = 30


def _float_param(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def _int_param(level, maxval):
    return int(level * maxval / PARAMETER_MAX)


def identity(img, _):
    return img


def auto_contrast(img, _):
    return ImageOps.autocontrast(img)


def equalize(img, _):
    return ImageOps.equalize(img)


def rotate(img, level):
    degrees = _float_param(level, 30.0)
    if random.random() > 0.5:
        degrees = -degrees
    return img.rotate(degrees)


def solarize(img, level):
    threshold = 256 - _int_param(level, 256)
    return ImageOps.solarize(img, threshold)


def color(img, level):
    magnitude = _float_param(level, 1.8) + 0.1
    if random.random() > 0.5:
        magnitude = 2.0 - magnitude
    return ImageEnhance.Color(img).enhance(magnitude)


def posterize(img, level):
    bits = 8 - _int_param(level, 4)
    return ImageOps.posterize(img, bits)


def contrast(img, level):
    magnitude = _float_param(level, 1.8) + 0.1
    if random.random() > 0.5:
        magnitude = 2.0 - magnitude
    return ImageEnhance.Contrast(img).enhance(magnitude)


def brightness(img, level):
    magnitude = _float_param(level, 1.8) + 0.1
    if random.random() > 0.5:
        magnitude = 2.0 - magnitude
    return ImageEnhance.Brightness(img).enhance(magnitude)


def sharpness(img, level):
    magnitude = _float_param(level, 1.8) + 0.1
    if random.random() > 0.5:
        magnitude = 2.0 - magnitude
    return ImageEnhance.Sharpness(img).enhance(magnitude)


def shear_x(img, level):
    magnitude = _float_param(level, 0.3)
    if random.random() > 0.5:
        magnitude = -magnitude
    return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))


def shear_y(img, level):
    magnitude = _float_param(level, 0.3)
    if random.random() > 0.5:
        magnitude = -magnitude
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))


def translate_x(img, level):
    magnitude = _int_param(level, img.size[0] / 3)
    if random.random() > 0.5:
        magnitude = -magnitude
    return img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0))


def translate_y(img, level):
    magnitude = _int_param(level, img.size[1] / 3)
    if random.random() > 0.5:
        magnitude = -magnitude
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude))


RA_SPACE = [
    identity, auto_contrast, equalize, rotate, solarize, color,
    posterize, contrast, brightness, sharpness,
    shear_x, shear_y, translate_x, translate_y,
]


class TrivialAugment:
    def __init__(self, augmentation_space=None, num_strengths=31):
        self.augmentation_space = augmentation_space or RA_SPACE
        self.num_strengths = num_strengths

    def __call__(self, img):
        op = random.choice(self.augmentation_space)
        strength = random.randint(0, self.num_strengths - 1)
        return op(img, strength)
