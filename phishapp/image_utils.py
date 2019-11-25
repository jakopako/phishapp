import argparse
import os
import cv2


def crop_image(filename):
    try:
        img = cv2.imread(filename)
        height, width, channels = img.shape
        margin = 50
        crop_img = img[0+margin:height, 0:width]
        new_filename = '{}_cr.png'.format(filename.rstrip('.png'))
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
        cv2.imwrite(new_filename, crop_img)
    except Exception:
        print("Couldn't crop file for some reason.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--crop", help="crop all png images in the given directory", action="store")
    args = parser.parse_args()

    if not args.crop:
        print("Please provide an image path.")
    else:
        for subdir, dirs, files in os.walk(args.crop):
            for file in files:
                if file.endswith('.png'):
                    if not file.endswith('_cr.png'):
                        full_path = os.path.join(subdir, file)
                        print('Cropping {}..'.format(full_path))
                        crop_image(full_path)