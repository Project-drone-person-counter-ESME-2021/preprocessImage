import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def trim(frame):
    # crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # crop top
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # crop top
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def concat_2_img(img, img2, img_, img1):

    """

    :param img: image left
    :param img2: image left cvt color
    :param img_: image riimg_leftght
    :param img1: image right cvt color
    :return: concat img of of the image
    """

    # function to detect key point on img
    sift = cv2.xfeatures2d.SIFT_create()
    # find key points
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # match key point in that are the same in the 2 img
    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    good = []

    print("number of matches between 2 photos : {}".format(len(matches)))

    for m, n in matches:
        # print("test m.distance({}) < 0.03 * n.distance({})".format(m.distance, 0.03 * n.distance))
        if m.distance < (0.3 * n.distance):
            good.append(m)

    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    else:
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)

        img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
        cv2.imshow("original_image_drawMatches.jpg", img3)

        print "Not enought matches are found - {} / {}, {}" \
              "".format(len(good), MIN_MATCH_COUNT, len(good) / MIN_MATCH_COUNT)
        plt.show()
        cv2.waitKey(0)
        quit()

    dst = cv2.warpPerspective(img_, M, (img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img

    return trim(dst)


def concat_list(_list_img, _list_img_cvt_color):
    img_left = _list_img[0]
    img_left_cvt_color = _list_img_cvt_color[0]
    for index_file_name in range(len(_list_img) - 1):
        # concat two image
        img_left = concat_2_img(img_left,
                                img_left_cvt_color,
                                _list_img[index_file_name + 1],
                                _list_img_cvt_color[index_file_name + 1]
                                )
        img_left_cvt_color = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

    # return all image concat
    return img_left


def convert_to_img_cv2_from_path(path_folder, name_file):
    img_ = cv2.imread(path_folder + name_file)
    img = cv2.resize(img_, (0, 0), fx=1, fy=1)
    img_cvt_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, img_cvt_color


def create_list_img(path_folder):
    # not sure see when the img from drone comes
    # get all name_files in folder
    list_name_files_img = sorted(os.listdir(path_folder))

    list_img = []
    list_img_cvt_color = []

    for name_img in list_name_files_img:
        # not sure see when img from drone comes
        # convert img from path to cv2 format
        img, img_cvt_color = convert_to_img_cv2_from_path(path_folder, name_img)

        # add to list
        list_img.append(img)
        list_img_cvt_color.append(img_cvt_color)

    print list_name_files_img
    return list_img, list_img_cvt_color


if __name__ == "__main__":
    list_img, list_img_cvt_color = create_list_img('./src/panorama_multiple/')

    output = concat_list(list_img, list_img_cvt_color)

    img_true = cv2.imread("./src/panorama/panorama_montains.jpeg")
    cv2.imshow("true", img_true)
    cv2.imshow("test", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
