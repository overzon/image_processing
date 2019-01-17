import numpy as np
import cv2

image = cv2.imread('p2.jpg')


(height,width,canel) = image.shape

gauss = (1.0/16) * np.array(
        [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])
sum(sum(gauss))

def gauss_bluer(image):
    img_out = image.copy()
    for i in np.arange(0, height-1):
        for j in np.arange(0, width-1):        
            sum = 0
            for k in np.arange(-1, 2):
                for l in np.arange(-1, 2):
                    a = image.item(i+k, j+l)
                    p = gauss[1+k, 1+l]
                    sum = sum + (p * a)
            b = sum
            img_out.itemset((i,j), b)
    return img_out

def mean_fliter(image):
    im = image
    for i in range(0,height-1):
        for j in range(0,width-1):
            convolutional = []
            for k in range(-1, 2):
                for l in range(-1, 2):
                    pix = image.item(i+k, j+l)
                    convolutional.append(pix)
            mean = sum(convolutional)/9
            im[i][j] = int(mean)
    return im

def median_filter(image):
    im = image
    for i in range(0,height-1):
        for j in range(0,width-1):
            neighbors = []
            for k in range(-1, 2):
                for l in range(-1, 2):
                    a = image.item(i+k, j+l)
                    neighbors.append(a)
            neighbors = sorted(neighbors)
            im[i][j] = neighbors[4]
    return im

def bgrtogray(image):
    # blue = [0] 7%# green = [1] 72%# red = [2] 0.299
    grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def backwhile(image):
    # blue = [0] 7%# green = [1] 72%# red = [2] 0.299
    # grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    # gray_img = grayValue.astype(np.uint8)
    img_out = image
    for i in range(0,height-1):
        for j in range(0,width-1):
            # print(image[i,j])
            if image[i,j] > 200:
                img_out[i,j] = 255
            else:
                img_out[i,j] = 0

    return img_out

img = bgrtogray(image)
cv2.imshow("gray",img)
cv2.imwrite("gray.jpg",img)

img_gauss = gauss_bluer(img)
cv2.imshow("gauss",img_gauss)
cv2.imwrite("gauss.jpg",img_gauss)

img_mean = mean_fliter(img)
cv2.imshow("mean",img_mean)
cv2.imwrite("mean.jpg",img_mean)

img_median = median_filter(img)
cv2.imshow("median",img_median)
cv2.imwrite("median.jpg",img_median)

bw = backwhile(img)
cv2.imshow("cc",bw)
cv2.imwrite("median.jpg",bw)

cv2.waitKey(0)
cv2.destroyAllWindows()