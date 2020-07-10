import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()


def match_points(
    ref_kp, ref_des,
    tar_kp, tar_des,
    min_match_count=10,
    flann_index_kdtree=0):
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(tar_des, ref_des, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > min_match_count:
            # 対応している点を返す
            src_pts = np.float32([ tar_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ ref_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            return src_pts, dst_pts
        else:
            return None, None


def detect_keypoint(reference):
    return sift.detectAndCompute(reference, None)


def match_image(
    reference, 
    target, 
    min_match_count=10,
    flann_index_kdtree=0):
        ref_kp, ref_des = detect_keypoint(reference)
        tar_kp, tar_des = detect_keypoint(target)
        return match_points(
            ref_kp, ref_des,
            tar_kp ,tar_des,
            min_match_count, flann_index_kdtree)


def get_homography(src_points, dst_points):
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H

