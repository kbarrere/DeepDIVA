from PIL import Image
import logging


class ResizeHeight(object):
    """
    Resize the image to a fixed height and keep the image ratio unchanged.
    """

    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, img):
        return resize_height(img, self.target_height)


def resize_height(img, target_height):
    width, height = img.size
    target_width = int(target_height * width / height)
    img = img.resize([target_width, target_height], Image.ANTIALIAS)
    return img


class PadRight(object):
    """
    Resize the image by to a fixed width.
    It keep the height unchanged and add zero for the value added.
    """

    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        return pad_right(img, self.target_width)


def pad_right(img, target_width, fill_color=(0, 0, 0)):
    width, height = img.size

    if width > target_width:
        logging.warning("Cannot pad an image of width " + str(width) + " to " + str(target_width))

    new_img = Image.new('RGB', (target_width, height), fill_color)
    new_img.paste(img, (0, 0))

    return new_img
