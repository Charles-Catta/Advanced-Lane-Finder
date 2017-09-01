import cv2
import warnings
import numpy as np


class ChessBoard(object):
    """ A basic ChessBoard object defining the parameters of the real life
        chessboard used for camera calibration
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.obj_points = self._create_obj_points_matrix()

    def get_corners(self, img):
        """ Get the chessboard corners in an image
        :param img: A numpy image of the chessboard
        :return: Corner coordinates in a numpy array
        :return: None if not all corners were in the image
        """
        got_corners, corners = cv2.findChessboardCorners(img, self.shape, None)

        if not got_corners:
            warnings.warn("Did not find all chessboard corners! \
                          is the chessboard correctly defined?",
                          RuntimeWarning)
            return None

        elif got_corners:
            return corners

    def _create_obj_points_matrix(self):
        """ Creates a generic object point matrix based on the chessboard size
        http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#calibratecamera
        :return: Object points numpy array
        """
        objp = np.zeros((self.y * self.x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.x, 0:self.y].T.reshape(-1, 2)
        return objp
