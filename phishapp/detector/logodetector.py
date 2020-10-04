import argparse
import cv2
import os
import numpy as np


class LogoDetector:

    def __init__(self, logo_path):
        self.logo_kps_desc = []
        # Patent has expired, see https://github.com/opencv/opencv/issues/16736
        # SIFT seems to work way better than ORB in this case.
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher.create()
        for filename in os.listdir(logo_path):
            logo_img = cv2.imread(os.path.join(logo_path, filename), cv2.IMREAD_GRAYSCALE)
            logo_kp, logo_des = self.sift.detectAndCompute(logo_img, None)
            logo_brand = filename.split('.')[0]  # logo files have to be named accordingly.
            self.logo_kps_desc.append((logo_brand, logo_img, logo_kp, logo_des))

    def find_logos(self, image, debug_level=0):
        """
        This method checks the given image against all logos whose keypoints have been extracted during the
        initialization of this class.
        :param image:
        :return:
        """
        logo_dict = {}
        for logo_brand, logo_img, logo_kp, logo_des in self.logo_kps_desc:
            img_kp, img_des = self.sift.detectAndCompute(image, None)
            matches = self.matcher.knnMatch(logo_des, img_des, k=2)
            # ratio test as per Lowe's paper https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
            good_matches = []
            for index in range(len(matches)):
                if len(matches[index]) == 2:
                    m, n = matches[index]
                    if m.distance < 0.75 * n.distance:
                        good_matches.append([m])
            if debug_level == 2:
                image_with_matches = cv2.drawMatchesKnn(logo_img, logo_kp, image, img_kp, good_matches, None)
                cv2.imshow("Matches of {} logo".format(logo_brand), image_with_matches)
                cv2.waitKey(0)
            if len(good_matches) < 0.13 * len(logo_kp):
                continue
            else:
                good_matches_points = np.float32([img_kp[m[0].trainIdx].pt for m in good_matches])
                # remove outliers if there are any
                good_matches_points = self.remove_outliers(good_matches_points)
                x, y, w, h = cv2.boundingRect(good_matches_points)
                if not self.correct_ratio(h, w, logo_img.shape[0], logo_img.shape[1]):
                    continue
                scale_factor = 2
                x_sc = max(int(x - ((scale_factor - 1) / 2) * w), 0)
                y_sc = max(int(y - ((scale_factor - 1) / 2) * h), 0)
                w_sc = min(w * scale_factor, image.shape[1] - x_sc)
                h_sc = min(h * scale_factor, image.shape[0] - y_sc)
                logo_dict[logo_brand] = (x_sc, y_sc, w_sc, h_sc)
                if debug_level >= 1:
                    tmp = np.copy(image)
                    cv2.rectangle(tmp, (x_sc, y_sc), (x_sc + w_sc, y_sc + h_sc), (0, 255, 0), 2)
                    cv2.rectangle(tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Screenshot with {} logo box".format(logo_brand), tmp)
                    cv2.waitKey(0)
        return logo_dict

    def find_logo(self, image, logo_image):
        """
        This method checks the whether the given logo can be found in the given image.
        :param image:
        :param logo_image:
        :return:
        """
        raise NotImplementedError

    def get_all_supported_brands(self):
        return [b for b, _, _, _ in self.logo_kps_desc]

    @staticmethod
    def open_image_from_path(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def remove_outliers(points):
        """
        A simple sliding window algorithm that checks whether every point has at least a minimum amount of close neighbours.
        Every point in points that doesn't is considered an outlier. The algorithm is far from perfect and there are cases
        where points would be considered outliers although they aren't. However, this way, the algorithm is supposed to be
        fast.
        :param points: an array of 2D points
        :return: the input array with all outliers removed
        """
        max_neighbour_points_check = 5  # on either side of the point to check
        min_neighbour_points = 3
        neighbour_radius_square = 20**2
        points = sorted(points, key=lambda p: (p[0], p[1]))
        clean_points = []
        for i in range(len(points)):
            neighbours_found = 0
            min_j = min(max(0, i - max_neighbour_points_check), len(points) - 2 * max_neighbour_points_check)
            max_j = min(i + max_neighbour_points_check - min(0, i - max_neighbour_points_check), len(points))
            for j in range(min_j, max_j, 1):
                if j == i:
                    continue
                distance_square = (points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2
                if distance_square <= neighbour_radius_square:
                    neighbours_found += 1
                    if neighbours_found >= min_neighbour_points:
                        clean_points.append(points[i])
                        break
        return np.float32(clean_points)

    @staticmethod
    def correct_ratio(h1, w1, h2, w2):
        # check if orientation if the same
        if h1 > w1:
            if h2 < w2:
                return False
        if h1 < w1:
            if h2 > w2:
                return False

        # TODO: check if ratio is similar
        #ratio_diff = abs(h1 / w1 - h2 / w2)
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="directory that contains the logos for the initialization",  action="store")
    parser.add_argument("-i", "--image",
                        help="the image where the logos shall be found",
                        action="store")
    parser.add_argument("-l", "--logo", help="a custom logo to look for in the image given by the -i option",
                        action="store")
    args = parser.parse_args()
    if not args.path:
        print("Please provide a path where the logos reside.")
    else:
        detector = LogoDetector(args.path)
        if args.logo:
            pass
        elif args.image:
            image = detector.open_image_from_path(args.image)
            ld = detector.find_logos(image, debug_level=1)
            print(ld)
