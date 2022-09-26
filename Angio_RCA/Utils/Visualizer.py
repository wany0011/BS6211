import numpy as np
import cv2
from PIL import Image


def draw_pts(img, coord, n_pts):

    # bgr_img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2BGR)
    # take RGB-image as input
    bgr_img = np.copy(img)

    pts_ = np.linspace(0, coord.shape[0]-1, num=n_pts, endpoint=True).astype(int)

    # print(pts_)

    for pt in pts_:

        row = coord[pt, 0]
        col = coord[pt, 1]

        if pt == 0:
            bgr_img[row-3:row+4, col, :] = [0, 0, 255]
            bgr_img[row, col-3:col+4, :] = [0, 0, 255]
        else:
            bgr_img[row-3:row+4, col, :] = [255, 0, 0]
            bgr_img[row, col-3:col+4, :] = [255, 0, 0]

    return bgr_img


def overlay_imgs(img, msk, coord, n_pts):
    # convert gray-scale to GRB
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    msk_img = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)

    labeled_img = draw_pts(bgr_img, coord, n_pts)
    mask_img = Image.fromarray(msk).convert("RGBA")

    # fg_img_trans = Image.new("RGBA", background.size)
    foreground = Image.fromarray(labeled_img).convert("RGBA")
    # fg_img_trans = Image.blend(fg_img_trans, labeled_img, 0.7)
    # foreground.paste(mask_img, (0, 0), mask_img)
    # Image.alpha_composite(foreground, mask_img)

    return Image.blend(mask_img, foreground, alpha=0.70)

