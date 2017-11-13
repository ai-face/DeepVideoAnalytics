
import cv2
import matplotlib.pyplot as plt

def showImgs(faces) :
    l = 1
    if isinstance(faces, list ):
        l=len(faces)
        l = min(10, l)
        fig, ax = plt.subplots(1,l)
        if( l > 1):
            for i in range(l):
                ax[i].xaxis.set_major_locator(plt.NullLocator())
                ax[i].yaxis.set_major_locator(plt.NullLocator())
                ax[i].imshow(cv2.cvtColor(faces[i], cv2.COLOR_BGR2RGB), cmap="bone")
        else:
            showImg(faces[0])
    else :
        showImg(faces)


def showImg(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="bone")

    # return fig