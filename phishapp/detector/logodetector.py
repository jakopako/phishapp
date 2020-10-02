# resources
# https://medium.com/open-knowledge/label-recognition-with-orb-da2b0260f4e4
# https://ai-facets.org/robust-logo-detection-with-opencv/


import cv2
import numpy as np


class LogoDetector:
    pass


orb = cv2.ORB_create(nfeatures=2000, edgeThreshold=20, patchSize=20)
logo_img = cv2.imread("../../logos/postfinance.png")
query_img = cv2.imread("../../screenshots/postfinance.png")

kp_logo, des_logo = orb.detectAndCompute(logo_img, None)
kp_query, des_query = orb.detectAndCompute(query_img, None)

result_logo_img = cv2.drawKeypoints(logo_img, kp_logo, None, flags=0)
result_query_img = cv2.drawKeypoints(query_img, kp_query, None, flags=0)


# FLANN parameters
flann_index_lsh = 6
index_params = dict(algorithm=flann_index_lsh,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2)
search_params = dict(checks=100)  # or pass empty dictionary

# create FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)

# perform matching
flann_matches = flann.knnMatch(des_logo, des_query, k=2)

# Need to draw only good matches, so create a mask
matches_mask = [[0, 0] for i in range(len(flann_matches))]

# ratio test as per Lowe's paper
good = []
for index in range(len(flann_matches)):
    if len(flann_matches[index]) == 2:
        m, n = flann_matches[index]
        if m.distance < 0.8 * n.distance:  # 0.8 is threshold of ratio testing
            matches_mask[index] = [1, 0]
            good.append(flann_matches[index])

draw_params = dict(
    singlePointColor=(255, 0, 0),
    matchesMask=matches_mask,
    flags=2)

img3 = cv2.drawMatchesKnn(logo_img, kp_logo, query_img, kp_query, flann_matches, None, **draw_params)

cv2.imshow("matching", img3)
cv2.imshow("logo image", result_logo_img)
cv2.imshow("query image", result_query_img)
cv2.waitKey(0)

