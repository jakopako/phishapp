# resources
# https://medium.com/open-knowledge/label-recognition-with-orb-da2b0260f4e4
# https://ai-facets.org/robust-logo-detection-with-opencv/


import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt


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


def get_features(img, method=1):
    if method == 1:
        orb = cv2.ORB_create(nfeatures=2000)
        return orb.detectAndCompute(img, None)
    elif method == 2:
        # Patent has expired, see https://github.com/opencv/opencv/issues/16736
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(img, None)


def cluster_points(X):
    db = DBSCAN(eps=100, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


class LogoDetector:
    pass


logo_img = cv2.imread("../../logos/dhl.png", cv2.IMREAD_GRAYSCALE)
query_img = cv2.imread("C:\\Users\\jakob\\data\\phishing\\validation\\DHL\\VIhHQItvEZ774Z5DhaQR1IlN.png", cv2.IMREAD_GRAYSCALE)
#query_img = cv2.imread("../../screenshots/netflix.png", cv2.IMREAD_GRAYSCALE)


kp_logo, des_logo = get_features(logo_img, method=2)
kp_query, des_query = get_features(query_img, method=2)


# FLANN parameters
flann_index_lsh = 6
index_params = dict(algorithm=flann_index_lsh,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2)
search_params = dict(checks=100)  # or pass empty dictionary

#matcher = cv2.FlannBasedMatcher(index_params, search_params)
matcher = cv2.BFMatcher.create()

# perform matching
matches = matcher.knnMatch(des_logo, des_query, k=2)

# Need to draw only good matches, so create a mask
matches_mask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good_matches = []
for index in range(len(matches)):
    if len(matches[index]) == 2:
        m, n = matches[index]
        if m.distance < 0.75 * n.distance:  # ratio test as per Lowe's paper https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
            matches_mask[index] = [1, 0]
            good_matches.append([m])

print(len(good_matches))
print(len(kp_logo))

draw_params = dict(
    singlePointColor=(255, 0, 0),
    matchesMask=matches_mask,
    flags=2)

img3 = cv2.drawMatchesKnn(logo_img, kp_logo, query_img, kp_query, good_matches, None)
cv2.imshow("matching", img3)
cv2.waitKey(0)
# if len(good_matches) < 0.13 * len(kp_logo):
#     print("No match found")
#     exit(0)

good_matches_points = np.float32([kp_query[m[0].trainIdx].pt for m in good_matches])
# remove outliers if there are any
good_matches_clusters = cluster_points(good_matches_points)
good_matches_points = remove_outliers(good_matches_points)

x, y, w, h = cv2.boundingRect(good_matches_points)
# TODO check, if ratio of bounding box corresponds more or less to logo ratio

# cv2.imshow("logo image", result_logo_img)

#box = cv2.boxPoints(rect)
#box = np.int0(box)
scale_factor = 2
x_sc = int(x - ((scale_factor - 1 )/ 2) * w)
y_sc = int(y - ((scale_factor - 1 )/ 2) * h)
w_sc = w * scale_factor
h_sc = h * scale_factor
cv2.rectangle(query_img, (x_sc, y_sc), (x_sc+w_sc,y_sc+h_sc), (0, 255, 0), 2)
cv2.rectangle(query_img, (x, y), (x+w,y+h), (0, 255, 0), 2)
cv2.imshow("query image", query_img)
cv2.waitKey(0)

