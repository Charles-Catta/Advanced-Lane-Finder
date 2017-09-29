import cv2
import numpy as np
from CameraInput import CameraInput


class LaneCamera(CameraInput):
    """
    A camera that can be calibrated and get a birds eye view
    """

    def __init__(self, camera_input):
        super().__init__(camera_input)
        self.calibrated = False
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.src_pts = np.float32([
            [230, 720],  # bottom left
            [575, 460],  # top left
            [715, 460],  # top right
            [1100, 720],  # bottom right
        ])
        self.dest_pts = np.float32([
            [480, 720],
            [410, 0],
            [960, 0],
            [920, 720]
        ])
        self.projection_matrix = cv2.getPerspectiveTransform(self.src_pts,
                                                             self.dest_pts)
        self.inverse_projection_matrix = np.linalg.inv(self.projection_matrix)

    def calibrate(self, calibration_imgs, chessboard):
        """Calibrates the camera based on the given ChessBoard object and images
        :param calibration_imgs: List of paths to calibration images
        :param chessboard: A ChessBoard object that defines the chessboard
                           used for calibration
        :type chessboard: ChessBoard object
        :returns: A tuple containing the camera matrix and distortion
                  coefficients
        """
        obj_points = []
        img_points = []

        for img in calibration_imgs:
            img = self.read_img_2_gray(img)
            corners = chessboard.get_corners(img)
            if corners is None:
                # OpenCV returns None if not all corners were in the image
                continue
            else:
                img_points.append(corners)
                obj_points.append(chessboard.obj_points)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            img.shape[::-1],
            None,
            None)

        self.camera_matrix = mtx
        self.distortion_coeffs = dist
        self.calibrated = True
        return (mtx, dist)

    def set_camera_calibration(self, camera_matrix, distortion_coeffs):
        """ Set the camera calibration if the camera matrix and distortion
            coefficients are already known
            :param camera_matrix: The camera matrix of the camera
            :param distortion_coeffs: The distortion coefficients of the lens
            :type camera_matrix: Numpy array
            :type distortion_coeffs: Numpy array
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.calibrated = True

    def save_camera_calibration(self, path):
        """ Save the camera calibration data to a .npy file
            :param path: Path to save the data to
        """
        assert self.calibrated, \
            "You need to calibrate the camera before saving the calibration"
        np.save(path,
                (self.camera_matrix, self.distortion_coeffs),
                allow_pickle=True)

    def load_camera_calibration(self, path):
        """ Load the camera calibration data from a .npy file
            :param path: Path to the .npy file
        """
        calibration = np.load(path)
        self.camera_matrix, self.distortion_coeffs = calibration
        self.calibrated = True

    def set_birds_eye_view_pts(self, src_pts, dest_pts):
        """ Modify the source and destination points used for the perspective transform
            to get a birds eye view of the road. The transform works by skewing
            the source points to their destination point
            :param src_pts: Source points of the transform
            :param dest_pts: Destination points of the transform
        """
        self.src_pts = src_pts
        self.dest_pts = dest_pts
        self.projection_matrix = cv2.getPerspectiveTransform(self.src_pts,
                                                             self.dest_pts)

    def _undistort(self, img):
        """ Undistort the image using the data found during calibration
        :param img: Image to undistort
        :return: Undistorted Image of the same size
        """
        assert self.calibrated, \
            "The camera should be calibrated with Camera.calibrate()"
        undist = cv2.undistort(img,
                               self.camera_matrix,
                               self.distortion_coeffs,
                               None,
                               self.camera_matrix)
        return undist

    def birds_eye_view(self, img):
        """ Get a birds eye view of the road
        :return: A birds eye view image of the road
        """
        img = self._undistort(img)
        img_y, img_x = img.shape[0:2]
        return cv2.warpPerspective(img, self.projection_matrix, (img_x, img_y))

    def inverse_birds_eye_view(self, img):
        """ Does the inverse perspective transform done in the
            birds_eye_view method
            :param img: Input image to unwarp
            :return: Unwarped image
        """
        img_y, img_x = img.shape[0:2]
        return cv2.warpPerspective(img, self.inverse_projection_matrix, (img_x, img_y))
