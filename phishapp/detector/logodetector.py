import argparse
import cv2
import os
import numpy as np
import base64


class LogoDetector:

    def __init__(self):
        self.logo_kps_desc = []
        # Patent has expired, see https://github.com/opencv/opencv/issues/16736
        # SIFT seems to work way better than ORB in this case.
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher.create()
        # FLANN parameters (FLANN seems to be slower than the BFMatcher for this use case and with these parameters.)
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)  # or pass empty dictionary
        # self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def load_logos(self, logo_path):
        """
        This method loads the logos from logo_path and calculates the keypoints and descriptors.
        :param logo_path: The path where the logos reside
        """
        for filename in os.listdir(logo_path):
            if filename.endswith('.png'):
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
        img_kp, img_des = self.sift.detectAndCompute(image, None)
        for logo_brand, logo_img, logo_kp, logo_des in self.logo_kps_desc:
            # calculate the two best matches for each feature descriptor
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
            if len(good_matches) < 0.25 * len(logo_kp):
                # If there are too few good matches we assume that the current logo does not appear in the given image.
                # This threshold has been found using trial and error.
                continue
            else:
                good_matches_points = np.float32([img_kp[m[0].trainIdx].pt for m in good_matches])
                # Remove outliers if there are any. This method is rather basic and could be improved by using some
                # kind of clustering and thereby also taking into account multiple occurrences of the same logo.
                good_matches_points = self.remove_outliers(good_matches_points)
                x, y, w, h = cv2.boundingRect(good_matches_points)
                if not self.correct_ratio(h, w, logo_img.shape[0], logo_img.shape[1]):
                    # If the shape of the bounding box does not approximately match that of the current logo we assume
                    # that it does not appear in the given image.
                    # TODO: instead of using width and height of the logo_img use the bounding box of the keypoints in
                    #  the logo_img.
                    continue
                # We scale the bounding box up a bit because keypoints tend to lie within the logo which means that the
                # box just fitting those points might be a bit too small.
                scale_factor = 1.5
                x_sc_tmp = int(x - ((scale_factor - 1) / 2) * w)
                y_sc_tmp = int(y - ((scale_factor - 1) / 2) * h)
                x_sc = max(x_sc_tmp, 0)
                y_sc = max(y_sc_tmp, 0)
                w_sc = min(int(w * scale_factor) + min(x_sc_tmp, 0), image.shape[1] - x_sc)
                h_sc = min(int(h * scale_factor) + min(y_sc_tmp, 0), image.shape[0] - y_sc)
                logo_dict[logo_brand] = (x_sc, y_sc, w_sc, h_sc)
                if debug_level >= 1:
                    tmp = np.copy(image)
                    cv2.rectangle(tmp, (x_sc, y_sc), (x_sc + w_sc, y_sc + h_sc), (0, 255, 0), 2)
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
        return sorted([b for b, _, _, _ in self.logo_kps_desc])

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
        neighbour_radius_square = 50**2
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
        err = 40
        if h1 > w1:
            if h2 < w2 - err:
                return False
        if h1 < w1:
            if h2 > w2 + err:
                return False

        # TODO: check if ratio is similar
        #ratio_diff = abs(h1 / w1 - h2 / w2)
        # print("h1: {}, h2: {}, w1: {}, w2: {}".format(h1, h2, w1, w2))
        # print("h1 / w1: {}".format(h1/w1))
        # print("h2 / h2: {}".format(h2/w2))
        # print("w1 / h1: {}".format(w1/h1))
        # print("w2 / h2: {}".format(w2/h2))
        return True

    @staticmethod
    def preprocess_image_from_base64(base64_string):
        decoded_data = base64.b64decode(base64_string)
        np_data = np.fromstring(decoded_data, np.uint8)
        return cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)


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
        detector = LogoDetector()
        detector.load_logos(args.path)
        if args.logo:
            pass
        elif args.image:
            image = detector.open_image_from_path(args.image)
            ld = detector.find_logos(image, debug_level=1)
            print(ld)
