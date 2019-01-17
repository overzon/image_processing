import cv2
import matplotlib.pyplot as plt
import numpy as np

global pixel ,equivalent_set
pixel = np.zeros(256)
equivalent_set = []

def bgrtogray(image):
    # blue = [0] 7%# green = [1] 72%# red = [2] 0.299
    grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def backwhile(image):
    (height,width) = image.shape
    img_out = image
    input1 = 50
    for i in range(0,height-1):
        for j in range(0,width-1):
            pixel[image[i,j]] += 1 
            if image[i,j] > input1:
                img_out[i,j] = 255
            else:
                img_out[i,j] = 0
    return img_out

def equivalent(lst):
    equ = []
    equ.append([0])
    for ll in range(len(lst)):
        check_same = 0
        for x in range(len(equ)):
            for i in lst[ll]:
                if i in equ[x]:
                    print (i, 'is in a')
                    # print("lst:",lst[ll],"  :  ",equ[x], end="")
                    equ[x] = np.union1d(equ[x],lst[ll])
                    # print("        ",equ[x] )
                    check_same = 1
                    break
        if check_same == 0:
            equ.append(lst[ll])
            # print("lst:",lst[ll],"  +  ",equ[x],"  =  ",equ[x+1])
            equ.append(lst[ll])
        print("")
    return equ

def new_blob_cut(i,j,new_blob,blob_cut,ob):
    if new_blob[i][j] == 0:
                blob_cut[i][j] = 255
    else:
        if new_blob[i][j] == ob:
            blob_cut[i][j] = 0
        else:
            blob_cut[i][j] = 255
    return blob_cut


def blob_cut_(equ,new_blob,ob):
    blob_cut = new_blob[:,:].copy()
    (height, width) = new_blob.shape
    for i in range(height):
        for j in range(width):
            for x in range(len(equ)):
                if new_blob[i][j] in equ[x]:
                    new_blob[i][j] = min(equ[x])
            new_blob_cut(i,j,new_blob,blob_cut,ob)
    return blob_cut

def list_equ(i,j,new_blob,count_):
    lst = []
    temp = [new_blob[i-1][j-1],new_blob[i-1][j],new_blob[i-1][j+1],new_blob[i][j-1]]
    if max(temp) == 0:
        count_ += 1
        new_blob[i][j] = count_
    else:
        temp = [x for x in temp if x !=0]
        new_blob[i][j] = min(temp)
        if len(np.unique(temp))>1:
            lst.append(np.unique(temp))
    return lst,count_

def blob_coloring(blob, ob):
    lst = []
    count_ = 0
    new_blob  = blob[:,:]
    height = len(new_blob)
    width = len(new_blob[0])
    for i in range(height):
        for j in range(width):
            if new_blob[i][j] == 255:
                new_blob[i][j] = 0
            else:
                new_blob[i][j] = 255
            # if(i == 0 or  j == 0 or j == width-1 ):
                # print("",end="")
            if new_blob[i][j] == 255:
                temp = [new_blob[i-1][j-1],new_blob[i-1][j],new_blob[i-1][j+1],new_blob[i][j-1]]
                if max(temp) == 0:
                    count_ += 1
                    new_blob[i][j] = count_
                else:
                    temp = [x for x in temp if x !=0]
                    new_blob[i][j] = min(temp)
                    if len(np.unique(temp))>1:
                        lst.append(np.unique(temp))
                # lst , count_ = list_equ(i,j,new_blob,count_)
    equ = equivalent(lst)
    blob_cut = blob_cut_(equ,new_blob,ob)
    re_img = blob.copy()
    for i in range(3):
        re_img[:,:] = blob_cut
    return  re_img


def for_loop(a,ra):
    if ra == 1:
        for i in range(len(a)):
            if 0 in a[i]:
                x = i
                return x
    else:
        for i in range(len(a)-1,-1,-1):
            if 0 in a[i]:
                x = i
                return x

def for_loop_i(a,ra):
    if ra == 1:
        for i in range(len(a[0])):
            if 0 in a[:,i]:
                x = i
                return x
    else:
        for i in range(len(a[0])-1,-1,-1):
            if 0 in a[:,i]:
                x = i
                return x

def crop_picture(img_for_crop):
    a = img_for_crop[:,:]
    top_left = for_loop(a,1)
    top_right = for_loop(a,0)
    bottom_left = for_loop_i(a,1)
    bottom_right = for_loop_i(a,0)
    high_crop = top_right-top_left+1
    with_crop = bottom_right-bottom_left+1
    crop_image = cv2.resize(img_for_crop, (with_crop, high_crop+10))
    print("shape_1",img_for_crop.shape,"shape_2",crop_image.shape)
    for i in range(top_left,top_right+1,1):
        for j in range(bottom_left,bottom_right,1):
            crop_image[i-top_left,j-bottom_left] = a[i][j]
    return crop_image


image = cv2.imread('train.png')
cv2.imshow("ori",image)

img = bgrtogray(image)
# cv2.imshow("gray",img)
# cv2.imwrite("gray.jpg",img)

bw = backwhile(img)
# cv2.imshow("blackwhile",bw)
# cv2.imwrite("blackwhile.jpg",bw)


blob = blob_coloring(bw,15)
cv2.imshow("blob",blob)
pic_ = crop_picture(blob)
cv2.imshow("crop_picture",pic_)
print(pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
