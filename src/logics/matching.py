import cv2
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()


def match_points(
    ref_kp, ref_des,
    tar_kp, tar_des,
    min_match_count=10,
    flann_index_kdtree=0):
    ref_des = np.float32(ref_des)
    tar_des = np.float32(tar_des)
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(tar_des, ref_des, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    if len(good) > min_match_count:
        # 対応している点を返す
        src_pts = np.float32([ tar_kp[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ ref_kp[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        return src_pts, dst_pts, good
    else:
        print(len(good))
        return None, None, good


def detect_keypoint(reference, algorithm=sift):
    return algorithm.detectAndCompute(reference, None)


def match_image(
    reference, 
    target, 
    min_match_count=10,
    flann_index_kdtree=0,
    algorithm=sift):
        ref_kp, ref_des = detect_keypoint(reference, algorithm)
        tar_kp, tar_des = detect_keypoint(target, algorithm)
        return match_points(
            ref_kp, ref_des,
            tar_kp ,tar_des,
            min_match_count, flann_index_kdtree)


def get_homography(src_points, dst_points):
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H


class MatcherBase:
    def set_from(self, from_target):
        pass

    def set_to(self, to_target):
        pass

    def match(self):
        pass


class PatternMatcher(MatcherBase):
    def __init__(self, min_match_count=10, flann_index_kdtree=0):
        self.min_match_count = min_match_count
        self.flann_index_kdtree = flann_index_kdtree
        self.akaze = cv2.AKAZE()

    def set_from(self, from_target):
        self.from_kp, self.from_des = self.akaze.detectAndCompute(from_target, None)

    def set_to(self, to_target):
        self.to_kp, self.to_des = self.akaze.detectAndCompute(to_target, None)

    def match(self):
        return match_points(
                self.ref_kp, self.ref_des,
                self.tar_kp, self.tar_des,
                self.min_match_count,
                self.flann_index_kdtree)


class TemplateMatcher(MatcherBase):
    def _get_description(self, target):
        # find Harris corners
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        return cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    def set_from(self, from_target):
        pass

    def set_to(self, to_target):
        pass

    def match(self):
        return match_points(
                self.ref_kp, self.ref_des,
                self.tar_kp, self.tar_des,
                self.min_match_count,
                self.flann_index_kdtree)


if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\Lab\Documents\sakuma\class\computervision\cv_finalreport\target.png")
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    print(centroids[0])


    