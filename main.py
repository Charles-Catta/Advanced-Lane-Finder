import cv2
import numpy as np
import glob
from Camera import LaneCamera
from ChessBoard import ChessBoard
from LaneDetector import LaneIsolator
from matplotlib import pyplot as plt

if __name__ == "__main__":
    lane_viewer = LaneCamera("project_video.mp4")
    lane_viewer.load_camera_calibration("camera_cal/calibration_data.npy")

    lane_detect = LaneIsolator()

    test_imgs = glob.glob("test_images/*.jpg")

    n_test_imgs = len(test_imgs)
    n_cols = 5
    plt.figure(figsize=(15, 10))

    """for idx, img in enumerate(test_imgs):
        img = lane_viewer.read_img(img)
        view = lane_viewer.birds_eye_view(img)
        color_selection = lane_detect.color_threshold(view)
        grad_selection = lane_detect.gradient_threshold(view)
        final_selection = lane_detect.isolate_lines(view)

        plt.subplot(n_test_imgs,
                    n_cols,
                    idx * n_cols + 1).set_title("Input", fontsize=8)
        plt.imshow(img)

        plt.subplot(n_test_imgs,
                    n_cols,
                    idx * n_cols + 2).set_title("Bird's eye view", fontsize=8)
        plt.imshow(view)

        plt.subplot(n_test_imgs,
                    n_cols,
                    idx * n_cols + 3).set_title("Color Selection", fontsize=8)
        plt.imshow(color_selection)

        plt.subplot(n_test_imgs,
                    n_cols,
                    idx * n_cols + 4).set_title("Gradient Selection", fontsize=8)
        plt.imshow(grad_selection)

        plt.subplot(n_test_imgs,
                    n_cols,
                    idx * n_cols + 5).set_title("Final Selection", fontsize=8)
        plt.imshow(final_selection)

    plt.show()"""

img = cv2.imread("test_images/straight_lines1.jpg")
cv2.imshow("image", img)
