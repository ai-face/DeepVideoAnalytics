

import cv2
import pyseeta
import os
import os.path

import rx
from rx import Observable as O

from pyseeta import Detector
from pyseeta import Aligner
from pyseeta import Identifier

class Seeta(object) :

    def __init__(self, model_path):
        # super(object, self).__init__()
        self.model_path = model_path
        #seeta_model_path = "/home/tony/ai_repo/model/seetaface"
        self.detector = Detector(os.path.join(self.model_path, "seeta_fd_frontal_v1.0.bin"))
        self.aligner = Aligner(os.path.join(self.model_path, "seeta_fa_v1.1.bin"))
        self.identifier = Identifier(os.path.join(self.model_path, "seeta_fr_v1.0.bin"))

        # self.detector.set_min_face_size(10)
        # self.detector.set_score_thresh(1.0)n


    def get_detector(self):
        return self.detector


    def get_aligner(self):
        return self.aligner


    def get_identifier(self):
        return self.identifier


    def detect(self, image):
        """
        :param image:
        :return: list[face], face = (face.top, face.left, face.right, face.bottom, face.score)
        """
        return self.detector.detect(image)


    def align(self, image, face):
        """
        :param image:
        :param face:
        :return: landmarks 5pts
        """
        return self.aligner.align(image, face)


    def crop(self, image_color):
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect(image_gray)
        if len(faces) == 0:
            return None
        landmarks = self.aligner.align(image_gray, faces[0])
        crop = self.identifier.crop_face(image_color, landmarks)
        return crop

    def crop_faces(self, image_color):
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect(image_gray)
        return [ self.identifier.crop_face(image_color,self.aligner.align(image_gray, f)) for f in faces]


    def crop_with_5pt(self, image, landmarks):
        return self.identifier.crop_face(image, landmarks)

    def crop_all(self, filepath):
        color = O.just(filepath).map(cv2.imread)
        clrpic = color.to_blocking().first()

        gry = color.map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
        grypic = gry.to_blocking().first()

        faces = gry.map(self.detect) \
            .flat_map(O.from_iterable) \
            .map(lambda x: self.align(grypic, x)) \
            .map(lambda x: self.crop_with_5pt(clrpic, x))

        return faces


    def feat(self, image):
        return self.identifier.extract_feature(image)


    def feat_with_5pt(self, image, landmarks):
        return self.identifier.extract_feature_with_crop(image, landmarks)

    def feat_crop(self, image_color):
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detect(image_gray)
        if len(faces) == 0:
            return None
        landmarks = self.aligner.align(image_gray, faces[0])
        # crop = self.identifier.crop_face(image_color, landmarks)

        return self.feat_with_5pt(image_color,landmarks)

    def similarity(self, featA, featB):
        return self.identifier.calc_similarity(featA, featB)