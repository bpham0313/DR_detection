from PIL import Image, ImageEnhance
from resizeimage import resizeimage
import os
import cv2


def resize_data(data_directory):
    root = os.listdir(data_directory)
    for item in root:
        with open(os.path.join(data_directory, item), 'r+b') as f:
            with Image.open(f) as image:
                print(item)
                rs = resizeimage.resize_cover(image, [512, 512])
                rs.save(os.path.join(data_directory, item), image.format)


def equalizer(data_directory):
    root = os.listdir(data_directory)
    for item in root:
        path = os.path.join(data_directory, item)
        img = cv2.imread(path)
        green = img[:, :, 1]
        #equ = cv2.equalizeHist(green)
        cv2.imwrite(path, equ)


def contrast_enhance(img):
    im = Image.open(img)
    enhancer_br = ImageEnhance.Brightness(im)
    enhancer_br.enhance(0.5).save("enhanced_img1.jpeg")
    enhancer_br.enhance(1.25).save("enhanced_img2.jpeg")


if __name__ == "__main__":

    data_dir = "test\\1"
    resize_data(data_dir)

    '''
    im='34445_right.jpeg'
    contrast_enhance(im)
    '''
