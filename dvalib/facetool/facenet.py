
from ..dlib.align_dlib import AlignDlib

class FaceNetActor(object) :

    def __int__(self):
        self.aligner = AlignDlib()


    def align(self, rgbImg, imgDim):
        return self.aligner.align(imgDim, rgbImg, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

