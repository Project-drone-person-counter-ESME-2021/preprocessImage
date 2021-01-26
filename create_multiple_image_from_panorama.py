from PIL import Image
import numpy as np
import random

"""
create multiple images from panorama
"""


# crop image at the right position and the right size
def crop_image(arr_img, width, pos_x, height=0, pos_y=0):
    new_img_arr = []

    if height == 0:
        new_img_arr = arr_img[:, pos_x:(width + pos_x)]

        print arr_img[:, pos_x:(width + pos_x)].shape
    # randomize height for better photos
    # else:
    #     height = random()

    return new_img_arr


# pre process image, get the image and crop it
def pre_process(path_folder_img, name_file_img, type_file_img, nb_img, randomize_width, randomize_height):
    # open image
    img = Image.open(path_folder_img + name_file_img + type_file_img)

    # convert to array
    img_array = np.array(img)

    print(img_array.shape)

    # calculate overlapping
    size_overlapping = 100
    total_overlapping = 0
    list_overlapping = [0]
    for i in range(nb_img):
        if not randomize_width:
            size_overlapping_for = size_overlapping

        else:
            size_overlapping_for = random.randrange(size_overlapping * 0.4, size_overlapping * 2)

        if i != 0:
            total_overlapping += size_overlapping_for
            list_overlapping.append(size_overlapping_for)

    # maybe put -1 to prevent going to far in the picture
    width_img = int((img_array.shape[1] + total_overlapping) / nb_img)

    print(width_img)

    list_crop_images = []
    pos_x = 0

    for i in range(nb_img):
        crop_img = crop_image(img_array, width_img, pos_x - list_overlapping[i])
        list_crop_images.append(crop_img)

        #print(crop_img.shape)

        pos_x += width_img - list_overlapping[i]
        print list_overlapping[i]
        img_final = Image.fromarray(crop_img, 'RGB')

        img_final.save('./src/panorama_multiple/' + name_file_img + "_photo_" + str(i) + ".jpg")


    index_img = 1


if __name__ == '__main__':
    pre_process("./src/panorama/", "panorama_montains", ".jpeg", 6, True, False)
