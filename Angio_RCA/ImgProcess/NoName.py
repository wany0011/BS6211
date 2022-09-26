import os
import cv2
import numpy as np

from ImgProcess.Img2Curve import mask2curve

'''automate find image files in a folder which are manually draw the curve. Move them to another directory, generate
concatenated image and npz'''


def main(total_pts):
    ip_dir = '/home/liuwei/Angio/Image/LAO_Main_Curved/New_train/'
    op_npz_dir = '/home/liuwei/Angio/Curve/MainCurve_LAO/Individual/New_train/'
    op_png_dir = '/home/liuwei/Angio/Image/LAO_Main_Curved/New_train/'

    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in file_list:

        if not os.path.exists(ip_dir+file):
            continue

        if file.startswith('.') or 'curved' not in file:
            continue

        id_ = '_'.join(file.split('_')[:2])    # for the other two
        print(file, id_)
        op_npz_name = op_npz_dir + id_ + '.npz'
        op_png_name = op_png_dir + id_ + '_curved_new.png'

        # if os.path.exists(op_npz_name):
        #     print('File exists.')
        #     continue
        combine_img = cv2.imread(ip_dir + id_ + '_curved.png', 0)

        print(combine_img.shape)
        resized_img = combine_img[:, :128]
        resized_mask = combine_img[:, -128:]

        print(resized_img.shape, resized_mask.shape)

        # resized_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)

        coord, bgr_mask = mask2curve(resized_mask, op_png_name, total_pts=total_pts)

        # img = cv2.imread('{}{}.png'.format(ip_dir, id_), 0)
        # print('!!!', ip_dir+file, np.max(img), ip_dir, file)
        # resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # print(img.shape, type(img))
        # label renamed as coord in augmented data
        np.savez_compressed(op_npz_name, img=resized_img, mask=resized_mask, coord=coord)

        print(np.max(resized_img), np.min(resized_img))
        bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        combine_img = np.concatenate((bgr_img, bgr_mask), axis=1)
        # print(bgr_img.shape)
        cv2.imwrite(op_png_name, combine_img)
        # command = 'rm {}{}*'.format(ip_dir, id_)
        # print(command)
        # os.system(command)


if __name__ == '__main__':
    main(total_pts=256)
