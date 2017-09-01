import cv2


class CameraInput(object):
    """ A base class for our Camera that integrates all image/video input methods
    """

    def __init__(self, video_input):
        """
        :param video_input: Input video file or a device
        """
        self.stream = cv2.VideoCapture(video_input)

    def __del__(self):
        self.stream.release()

    def get_frame(self):
        """ Get a single video frame
        :return: One frame of the video
        """
        ret, frame = self.stream.read()
        if ret:
            return frame

    def read_img(self, input_img):
        """ Reads an image from a path to a numpy array in BGR
        :param input_img: Path to the image
        :type input_img: String
        :return: A numpy array image
        """
        return cv2.imread(input_img)

    def read_img_2_gray(self, input_img):
        """ Reads an image from path into grayscale
        :param input_img: Path to image
        :type input_img: Path to image
        :return: A grayscale numpy array image
        """
        img = self.read_img(input_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
