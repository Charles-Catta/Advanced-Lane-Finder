import cv2
import numpy as np
import glob
from Camera import LaneCamera
from ChessBoard import ChessBoard
from LaneIsolator import LaneIsolator
from SlidingWindowDetector import SlidingWindowDetector
from matplotlib import pyplot as plt


if __name__ == "__main__":
    lane_viewer = LaneCamera("test_videos/challenge_video.mp4")
    lane_viewer.load_camera_calibration("camera_cal/calibration_data.npy")
    view = lane_viewer.get_frame()
    view = lane_viewer.birds_eye_view(view)
    lane_detect = LaneIsolator()
    lanes = lane_detect.isolate_lines(view)
    detector = SlidingWindowDetector()

    left_fit, right_fit = detector.get_lane_poly(lanes)
    plot_y = np.linspace(0, lanes.shape[0] - 1, lanes.shape[0])
    left_fit_x = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y**2 + \
        right_fit[1] * plot_y + right_fit[2]

    fig = plt.subplot(131)
    plt.plot(left_fit_x, plot_y)
    plt.plot(right_fit_x, plot_y)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.imshow(view)

    plt.subplot(132)
    plt.imshow(lanes)
    plt.subplot(133)
    plt.imshow(lane_viewer.inverse_birds_eye_view(view))
    plt.show()
