#Elif Caliskan 2016400183
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import sys

im = Image.open(sys.argv[1])
K = int(sys.argv[2])
mode = int(sys.argv[3])
plt.imshow(im)

def quantize(im, K):
    width, height = im.size

    points = np.zeros(K*2).reshape((K,2))
    if mode == 0:   #points are taken from click
        points = plt.ginput(K, show_clicks=True)
    else:  #points are found by random.uniform
        for i in range(K):
            points[i][0] = np.random.uniform(0, width)
            points[i][1] = np.random.uniform(0, height)

    #pixel_values is the rgb values of each pixel in image
    pixel_values = list(im.getdata())
    pixel_values = np.array(pixel_values).reshape((height, width, 3))
    # y represents row(height) and x represents column(width)
    cluster_rgbs = np.array([pixel_values[(int)(points[0][1])][(int)(points[0][0])]])
    #cluster_rgbs store rgb values of each pixel
    for i in range (K-1):
        cluster_rgbs = np.append(cluster_rgbs, np.array([pixel_values[(int)(points[i+1][1])][(int)(points[i+1][0])]]), axis = 0)

    #cluster_ids store the cluster id of each pixel in image
    cluster_ids = np.zeros(width * height)
    #cluster_distances store the distance of every pixel to each cluster
    cluster_distances = np.zeros(width*height*K).reshape((K,width*height))
    for a in range(10):
        for i in range (K):
            #the clusters rgs is subtracted from every pixel's rgs
            pixel_array = np.subtract(pixel_values, cluster_rgbs[i])
            pixel_array = np.power(pixel_array, 2)
            pixel_array = np.reshape(np.array(pixel_array), (width*height,3))
            #the distance is added to cluster_distances
            sums = np.sum(pixel_array, axis=1)
            cluster_distances[i] = sums

        #cluster id is found by finding the min distance
        cluster_ids = np.argmin(cluster_distances, axis=0)
        cluster_ids = np.reshape(cluster_ids, (height,width))

        #cluster_sums is used for the mean rgb value
        cluster_sums = np.zeros(K*3).reshape((K,3))
        cluster_counts = np.zeros(K)

        #iterate through every pixel and add to the related cluster sum
        for i in range(height):
            for j in range(width):
                cluster_sums[cluster_ids[i][j]] = np.add(cluster_sums[cluster_ids[i][j]], pixel_values[i][j])
                cluster_counts[cluster_ids[i][j]] += 1
        #rgb values of each cluster is found by taking the mean
        for i in range(K):
            cluster_rgbs[i] = np.divide(cluster_sums[i], cluster_counts[i])
    #each pixel's rgb is changed with corresponding cluster's rgb
    for i in range(height):
        for j in range(width):
            pixel_values[i][j] = cluster_rgbs[cluster_ids[i][j]]
    #image is formed by using the pixel array
    img = Image.fromarray(pixel_values.astype('uint8'), 'RGB')
    img.save("image.jpg")
    return img

quantize(im,K)
