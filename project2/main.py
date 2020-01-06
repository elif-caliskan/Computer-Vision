#Elif Çalışkan 2016400183
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image
import numpy as np
import operator
import sys

#this method was not used in solution but I implemented the normalization part
def normalize(points):
    normPoints = np.zeros(K * 3).reshape((K, 3))
    avg = np.sum(points, axis = 0)
    avg = np.true_divide(avg, len(points))
    print(avg)
    #for finding normalization matrix, average distance is calculated
    points_tmp = np.subtract(points, avg)
    points_tmp = np.power(points_tmp, 2)
    distances = np.sum(points, axis = 1)
    xy_norm = np.mean(distances)
    diagonal_element = np.sqrt(2) / xy_norm
    element_13 = -np.sqrt(2) * avg[0] / xy_norm
    element_23 = -np.sqrt(2) * avg[1] / xy_norm
    normalizationMatrix = [[diagonal_element, 0, element_13], [0, diagonal_element, element_23], [0, 0, 1]]
    for index in range(K):
        arr = [points[index][0], points[index][1], 1]
        normPoints[index] = np.matmul(normalizationMatrix, arr)
    return normPoints

#homography matrix is found with the helper matrix from slides
def computeH(im1Points, im2Points):
    helper = np.zeros(K * 2 * 9).reshape((2*K, 9))
    for i in range (K):
        helper[2*i] = np.array([im1Points[i][0]* im2Points[i][2], im1Points[i][1]* im2Points[i][2], im1Points[i][2]* im2Points[i][2], 0,0,0,
                                -im1Points[i][0]* im2Points[i][0],- im1Points[i][1] * im2Points[i][0], - im1Points[i][2] * im2Points[i][0]])
        helper[2 * i + 1] = np.array([0, 0, 0, im1Points[i][0] * im2Points[i][2], im1Points[i][1] * im2Points[i][2], im1Points[i][2] * im2Points[i][2],
             -im1Points[i][0] * im2Points[i][1], - im1Points[i][1] * im2Points[i][1], - im1Points[i][2] * im2Points[i][1]])
    #svd is used for getting singular vector
    u, s, v = np.linalg.svd(helper)
    H = v[-1, :]
    return H.reshape(3, 3)

# For warping, I was inspired by the function from: https://github.com/jmlipman/LAID/blob/master/IP/homography.py
def warp(image, H):
    width, height = image.size
    pixel_values = list(image.getdata())
    pixel_values = np.array(pixel_values).reshape((height, width, 3))
    #first, the corners are found by multiplying original corners with the homography matrix and finding x and y values by dividing them into k
    bunchX = []
    bunchY = []
    tt = np.array([[1],[1],[1]])
    tmp = np.dot(H, tt)
    tmp = np.divide(tmp, tmp[2])
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    tt = np.array([[width], [1], [1]])
    tmp = np.dot(H, tt)
    tmp = np.divide(tmp, tmp[2])
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    tt = np.array([[1], [height], [1]])
    tmp = np.dot(H, tt)
    tmp = np.divide(tmp, tmp[2])
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    tt = np.array([[width], [height], [1]])
    tmp = np.dot(H, tt)
    tmp = np.divide(tmp, tmp[2])
    bunchX.append(tmp[0])
    bunchY.append(tmp[1])

    #width = refX2 - refX1
    #height = refY2 - refY1
    refX1 = int(np.min(bunchX))
    refX2 = int(np.max(bunchX))
    refY1 = int(np.min(bunchY))
    refY2 = int(np.max(bunchY))

    # a result array is created
    result = np.zeros((int(refY2 - refY1), int(refX2 - refX1), 3))

    # transform evey pixel
    for i in range(width):
        for j in range(height):

            tt = np.array([i, j, 1])
            tmp = np.dot(H, tt)
            tmp = np.divide(tmp, tmp[2])
            x1 = int(tmp[0]) - refX1
            y1 = int(tmp[1]) - refY1

            if x1 > 0 and y1 > 0 and y1 < refY2 - refY1 and x1 < refX2 - refX1:
                result[y1, x1, :] = pixel_values[j][i]

    #after transforming each pixel, interpolation should be made
    #if the pixel is black, its rgb value is found by the original point from that image with the help of inverse matrix
    inverse = np.linalg.inv(H)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if sum(result[i, j, :]) == 0:
                tt = np.array([[j + refX1], [i + refY1], [1]])
                tmp = np.dot(inverse, tt)
                x1 = int(tmp[0] / tmp[2])
                y1 = int(tmp[1] / tmp[2])

                if x1 > 0 and y1 > 0 and x1 < width and y1 < height:
                    result[i, j, :] = pixel_values[y1, x1, :]
    return result

# overlap function shifts the first image and puts that image on top of the original one
# then, by using image.blend, these images are blended with alpha 0.5
def overlap(im1, im2):
    width1, height1 = im1.size
    width2, height2 = im2.size
    # shift value is found by getting the correspondance point of one point from the original image and finding the difference between those points
    # also there is an offset because of the size difference
    inverse = np.linalg.inv(homography)
    point1 = pointsMiddle[0]
    point2 = np.matmul(inverse, pointsMiddle[0])
    point2 = np.divide(point2, point2[2])

    shift = (int(point2[0] - point1[0] + (width1 -width2)/2), int(point2[1] - point1[1] + (height1 - height2)/2))

    # the size of the panorama
    nw, nh = map(max, map(operator.add, im2.size, shift), im1.size)
    print("nw ",nh, " nh ",nh)

    # paste im1 on top of im2
    newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg1.paste(im2, shift)
    newimg1.paste(im1, (0, 0))

    # paste im2 on top of im1
    newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg2.paste(im1, (0, 0))
    newimg2.paste(im2, shift)

    #the result is found by taking the average of these images
    result = Image.blend(newimg1, newimg2, alpha=0.5)
    result.save("imageOver.png")


im1 = Image.open(sys.argv[0])
im2 = Image.open(sys.argv[1])
K = int(sys.argv[2])
plt.imshow(im1)

pointsLeft = np.zeros(K*2).reshape((K, 2))
pointsLeft = plt.ginput(K, show_clicks=True)
pointsLeft2 = np.zeros(K*3).reshape((K, 3))
for m in range(K):
    pointsLeft2[m] = [pointsLeft[m][0], pointsLeft[m][1], 1]
print(pointsLeft)
plt.close()
plt.imshow(im2)

pointsMid= np.zeros(K*2).reshape((K, 2))
pointsMid = plt.ginput(K, show_clicks=True)
pointsMiddle = np.zeros(K*3).reshape((K, 3))
for m in range(K):
    pointsMiddle[m] = [pointsMid[m][0], pointsMid[m][1], 1]
print(pointsMiddle)
plt.close()

homography = computeH(pointsLeft2, pointsMiddle)
final = warp(im1, homography)
img = Image.fromarray(final.astype('uint8'), 'RGB')
img.save("image.jpg")
overlap(img, im2)







