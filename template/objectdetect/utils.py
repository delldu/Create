import random
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pdb

class Box(object):
    """Box Interpreter, (c,r) format, [[x1, y1] --> [x2, y2])."""

    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3

    def __init__(self, x1, y1, x2, y2):
        """Create Box."""
        self.data = [x1, y1, x2, y2]

    def left(self):
        """Box left."""
        return self.data[self.LEFT]

    def right(self):
        """Box right."""
        return self.data[self.RIGHT]

    def top(self):
        """Box top."""
        return self.data[self.TOP]

    def bottom(self):
        """Box bottom."""
        return self.data[self.BOTTOM]

    def height(self):
        """Box height."""
        return self.data[self.BOTTOM] - self.data[self.TOP]

    def width(self):
        """Box width."""
        return self.data[self.RIGHT] - self.data[self.LEFT]

    def xcenter(self):
        """Return x center."""
        return (self.data[self.LEFT] + self.data[self.RIGHT]) / 2

    def ycenter(self):
        """Return y center."""
        return (self.data[self.TOP] + self.data[self.BOTTOM]) / 2

    @classmethod
    def fromlist(cls, l):
        """Create box from list."""
        b = Box(l[0], l[1], l[2], l[3])
        # print("l:", l, "Box:", b)
        return b

    def tolist(self):
        """Transform box to list."""
        return self.data

    def __repr__(self):
        """Dump box."""
        return '(Box top: %d, left: %d, bottom: %d, right: %d)' % (
            self.data[self.TOP], self.data[self.LEFT], self.data[self.BOTTOM],
            self.data[self.RIGHT])


def random_colors(nums, bright=True, shuffle=True):
    """Generate colors from HSV space to RGB."""
    brightness = 1.0 if bright else 0.7
    hsv = [(i / nums, 1, brightness) for i in range(nums)]
    fcolors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = []
    for (r, g, b) in fcolors:
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    if shuffle:
        random.shuffle(colors)
    return colors


def blend_mask(image, mask, color):
    """Blend mask. image: PIL format, mask is tensor HxW. color: (r, g, b)."""
    colorimg = Image.new("RGB", image.size, color)
    mask = mask.numpy() * 255.0
    mask = mask.astype(np.uint8)
    # pdb.set_trace()

    maskimg = Image.fromarray(mask[0]).convert('L')

    # border is also mask image
    border = maskimg.filter(ImageFilter.CONTOUR).point(lambda i: 255 - i)
    # Image filter bug ?
    # remove border points on left == 0, right == 0, top == 0 and bottom == 0
    pixels = border.load()
    for i in range(border.height):
        pixels[0, i] = 0
        pixels[border.width - 1, i] = 0
    for j in range(border.width):
        pixels[j, 0] = 0
        pixels[j, border.height - 1] = 0

    alphaimg = Image.blend(image, colorimg, 0.2)
    context = Image.composite(alphaimg, image, maskimg)

    # Add border
    return Image.composite(colorimg, context, border)


def blend_image(image, label_names, boxes, masks, scores=None):
    """Blend image with label_names, boxes Nx4, masks NxHxW."""
    m = boxes.size(0)
    if m < 1:
        return image
    colors = random_colors(m)
    fusion = image
    if masks is not None:
        for i in range(masks.size(0)):
            fusion = blend_mask(fusion, masks[i], colors[i])

    draw = ImageDraw.Draw(fusion)
    for i in range(m):
        b = Box.fromlist(boxes[i].tolist())
        draw.rectangle((b.left(), b.top(), b.right(), b.bottom()), None, colors[i])
        label = label_names[i] if isinstance(label_names, list) else ""
        if scores is not None:
            label += " {:.3f}".format(scores[i])
        draw.text((b.left(), b.top()), label, colors[i])
    del draw
    return fusion
