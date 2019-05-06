import numpy as np
import matplotlib.pyplot as plot
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
import loadData as data
import scipy.misc

def getDatumImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def displayData(indices_to_display=None):
    X = np.insert(data.X,0,1,axis=1)

    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices_to_display:
        indices_to_display = random.sample(range(X.shape[0]), nrows * ncols)

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = getDatumImg(X[idx])
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    fig = plot.figure(figsize=(5, 5))
    img = big_picture
    plot.imshow(img, cmap=cm.Greys_r)
    plot.axis('off')




    plot.show()


def displayHiddenLayer(myTheta):

    # remove bias unit:
    myTheta = myTheta[:, 1:]
    assert myTheta.shape == (25, 400)

    width, height = 20, 20
    nrows, ncols = 5, 5

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0
    for row in myTheta:
        if icol == ncols:
            irow += 1
            icol = 0
        # add bias unit back in?
        iimg = getDatumImg(np.insert(row, 0, 1))
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    fig = plot.figure(figsize=(6, 6))
    img = scipy.misc.toimage(big_picture)
    plot.imshow(img, cmap=cm.Greys_r)
    plot.axis('off')
    plot.show()

def main():
	displayData()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
