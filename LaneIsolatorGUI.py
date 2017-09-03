import cv2
import numpy as np
from LaneIsolator import LaneIsolator


class LaneIsolatorGUI(LaneIsolator):
    def __init__(self, example_img):
        self.example_img = example_img
        self.color_thresh_img = np.zeros_like(example_img[0])
        self.grad_thresh_img = np.zeros_like(example_img[0])
        self.L_THRESHOLD = (75, 255)  # L in LUV color space
        self.S_THRESHOLD = (85, 235)  # S in HLS color space
        self.B_THRESHOLD = (100, 220)  # B in Lab color space
        self.MAG_THRESHOLD = (40, 215)  # Gradient magnitude
        self.DIR_THRESHOLD = (0, np.pi / 7)  # Gradient direction

    def calibrate(self):
        cv2.namedWindow("calibration")
        cv2.createTrackbar("L", "calibration", 0, 255, self.draw)
        cv2.createTrackbar("S", "calibration", 0, 255, self.draw)
        cv2.createTrackbar("B", "calibration", 0, 255, self.draw)
        cv2.createTrackbar("Magnitude", "calibration", 0, 255, self.draw)
        cv2.createTrackbar("Direction", "calibration", 0, 2 * np.pi, self.draw)
        while(1):
            cv2.imshow("calibration", self.example_img)
            cv2.imshow("calibration", self.color_thresh_img)
            cv2.imshow("calibration", self.grad_thresh_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        self._print_internal_state()
        cv2.destroyAllWindows()

    def draw_example_image(self):
        color = self.color_thresh_img = self.color_threshold(self.example_img)
        grad = self.grad_thresh_img = self.gradient_threshold(self.example_img)
        return (color, grad)

    def draw(self):
        L = cv2.getTrackbarPos("L", "calibration")
        S = cv2.getTrackbarPos("S", "calibration")
        B = cv2.getTrackbarPos("B", "calibration")
        Mag = cv2.getTrackbarPos("Magnitude", "calibration")
        Dir = cv2.getTrackbarPos("Direction", "calibration")
        self.set_calibration(L, S, B, Mag, Dir)
        self.color_thresh_img, self.grad_thresh_img = self.draw_example_image()

    def _print_internal_state(self):
        print(self.L_THRESHOLD)
        print(self.S_THRESHOLD)
        print(self.B_THRESHOLD)
        print(self.MAG_THRESHOLD)
        print(self.DIR_THRESHOLD)
