import cv2
import numpy as np


class SlidingWindowDetector(object):
    """ Implements the sliding window approach to find and fit lane lines
        from the isolated lane images
    """

    def __init__(self,
                 video_stream=True,
                 n_windows=10,
                 win_margin=100,
                 win_min_px=50):
        self.video_stream = video_stream
        self.got_first_frame = False
        self.n_windows = n_windows
        self.win_margin = win_margin
        self.win_min_px = win_min_px
        self.prev_left_lane = None
        self.prev_right_lane = None
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def get_lane_poly(self, img):
        """ Function that fits detected lane lines to polynomials
            :param img: Image of the isolated lane line pixels
            :return: Array of 2 polynomials
        """
        #if (self.got_first_frame == False):
        left_x, left_y, right_x, right_y = self._first_search(img)
        #    self.got_first_frame = True
        #else:
        #    left_x, left_y, right_x, right_y = self._next_frame_search(img)
        #    if len(left_x) == 0 or len(right_x) == 0:
        #        left_x, left_y, right_x, right_y = self._first_search(img)
        
        left_fit_m = np.polyfit(left_y * self.ym_per_pix, left_x * self.xm_per_pix, 2)
        right_fit_m = np.polyfit(right_y * self.ym_per_pix, right_x * self.xm_per_pix, 2)
        
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        
        self.prev_left_lane = left_fit
        self.prev_right_lane = right_fit
        
        return (left_fit, right_fit, left_fit_m, right_fit_m)
        
    def _first_search(self, img):
        """ Does a full sliding window search over the image, usually
            only for the first frame of a video
            :param img: Image of the isolated lane line pixels
            :return: Tuple containing the arrays of polynomials and the
                     array of pixels belonging to both left and right lanes
        """
        histogram = np.sum(img[np.int(img.shape[0] / 2):, :], axis=0)
        hist_mid = np.int(histogram.shape[0] / 2)
        hist_left_lane = np.argmax(histogram[:hist_mid])
        hist_right_lane = np.argmax(histogram[hist_mid:]) + hist_mid
        self.win_height = np.int(img.shape[0] / self.n_windows)

        non_zero = img.nonzero()
        non_zero_x = np.array(non_zero[1])
        non_zero_y = np.array(non_zero[0])

        left_lane = hist_left_lane
        right_lane = hist_right_lane

        left_idxs = []
        right_idxs = []

        for window in range(self.n_windows):
            win_y_low = img.shape[0] - (window + 1) * self.win_height
            win_y_high = img.shape[0] - window * self.win_height
            win_x_left_low = left_lane - self.win_margin
            win_x_left_high = left_lane + self.win_margin
            win_x_right_low = right_lane - self.win_margin
            win_x_right_high = right_lane + self.win_margin
            good_left_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                              (non_zero_x >= win_x_left_low) & (non_zero_x < win_x_left_high)).nonzero()[0]
            good_right_inds = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) &
                               (non_zero_x >= win_x_right_low) & (non_zero_x < win_x_right_high)).nonzero()[0]
            # Append these indices to the lists
            left_idxs.append(good_left_inds)
            right_idxs.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.win_min_px:
                left_lane = np.int(np.mean(non_zero_x[good_left_inds]))
            if len(good_right_inds) > self.win_min_px:
                right_lane = np.int(np.mean(non_zero_x[good_right_inds]))

        left_idxs = np.concatenate(left_idxs)
        right_idxs = np.concatenate(right_idxs)

        left_x = non_zero_x[left_idxs]
        left_y = non_zero_y[left_idxs]
        right_x = non_zero_x[right_idxs]
        right_y = non_zero_y[right_idxs]
        return (left_x, left_y, right_x, right_y)

    def _next_frame_search(self, img):
        """ Does a fuzzy sliding window search close to the last detected lane lines
            only works after having done a _full_search
            :param img: Image of the isolated lane line pixels
            :return: Tuple containing the arrays of polynomials and the
                     array of pixels belonging to both left and right lanes
        """
        assert self.got_first_frame, \
            "You need to do a full sliding window search first"
        non_zero = img.nonzero()
        non_zero_x = np.array(non_zero[1])
        non_zero_y = np.array(non_zero[0])
        left_lane_inds = ((non_zero_x > (self.prev_left_lane[0] * (non_zero_y**2) + self.prev_right_lane[1] * non_zero_y +
                                         self.prev_left_lane[2] - self.win_margin)) & (non_zero_x < (self.prev_left_lane[0] * (non_zero_y ** 2) +
                                                                                                     self.prev_left_lane[1] * non_zero_y + self.prev_left_lane[2] + self.win_margin)))

        right_lane_inds = ((non_zero_x > (self.prev_right_lane[0] * (non_zero_y**2) + self.prev_right_lane[1] * non_zero_y +
                                          self.prev_right_lane[2] - self.win_margin)) & (non_zero_x < (self.prev_right_lane[0] * (non_zero_y**2) +
                                                                                                       self.prev_right_lane[1] * non_zero_y + self.prev_right_lane[2] + self.win_margin)))

        # Extract left and right pixel positions
        left_x = non_zero_x[left_lane_inds]
        left_y = non_zero_y[left_lane_inds]
        right_x = non_zero_x[right_lane_inds]
        right_y = non_zero_y[right_lane_inds]
        return (left_x, left_y, right_x, right_y)
