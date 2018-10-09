from PIL import Image


class ResizeHeight(object):
    """
    Resize the image to a fixed height and keep the image ratio unchanged
    """

    def __call__(self, img):
        return resize_height(img)


def resize_height(img):
    width, height = img.size
    target_height = 128
    target_width = int(target_height * width / height)
    img = img.resize([target_width, target_height], Image.ANTIALIAS)
    return img
