import cv2
import hashlib
import numpy as np
import matplotlib.pyplot as plt


def convert_bgr2rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def pil2cv(image_pil, convert_channel=True):
    image_np = np.array(image_pil, dtype=np.uint8)
    if convert_channel:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_np


def show_image_from_ndarray(image_array, figsize=None, bgr2rgb=True):
    if figsize is not None:
        plt.figure(figsize=figsize)

    if (len(image_array.shape) < 3) or (image_array.shape[-1] == 1):
        plt.imshow(image_array)
        plt.axis("off")
    else:
        if bgr2rgb:
            plt.imshow(convert_bgr2rgb(image_array))
        else:
            plt.imshow(image_array)
        plt.axis("off")


def draw_multiple_image(titles, images, bgr2rgb=True, figsize=None):
    n_images = len(images)
    assert len(titles) == n_images

    if figsize is None:
        fig, axes = plt.subplots(1, n_images)
    else:
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
    for i in range(n_images):
        if bgr2rgb:
            axes[i].imshow(convert_bgr2rgb(images[i]))
        else:
            axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")


def generate_unique_color(uuid):
    # Calculate the MD5 hash of the UUID
    hash_object = hashlib.md5(uuid.encode())
    hash_hex = hash_object.hexdigest()

    # Take the first 6 characters of the hash and convert it to an integer
    color_int = int(hash_hex[:6], 16)

    # Convert the integer to RGB values
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF

    return r, g, b
