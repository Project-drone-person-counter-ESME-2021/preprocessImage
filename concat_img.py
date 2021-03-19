from function import concat_2_img

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


def concat_list(_list_img, _list_img_cvt_color):
    img_left = _list_img[0]
    img_left_cvt_color = _list_img_cvt_color[0]

    list_concat = []

    for index_file_name in range(len(_list_img) - 1):
        # concat two image
        verif, img = concat_2_img(img_left,
                                img_left_cvt_color,
                                _list_img[index_file_name + 1],
                                _list_img_cvt_color[index_file_name + 1]
                                )

        if verif:
            img_left = img
            img_left_cvt_color = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        else:
            list_concat.append(img_left)
            img_left = _list_img[index_file_name + 1]
            img_left_cvt_color = _list_img_cvt_color[index_file_name + 1]


    list_concat.append(img_left)

    # return all image concat
    return list_concat


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
    list_img, list_img_cvt_color = create_list_img('./src/test/')

    output = concat_list(list_img, list_img_cvt_color)
    index = 1
    for img in output:
        # img_true = cv2.imread("./src/panorama/panorama_montains.jpeg")
        #cv2.imshow("true", img_true)
        cv2.imshow("test " + str(index), img)
        index += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
