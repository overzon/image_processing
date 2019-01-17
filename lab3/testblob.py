import numpy as np
import cv2
from matplotlib import pyplot as plt
global min_equ
min_equ = []
def rgb_to_gray(img, chanel):
        new_img = img
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])
        #Weight
        if(chanel == 1):
                Avg = 0.299*R+0.587*G+0.114*B
        if(chanel == 2):
                Avg = 0.2126*R+0.7152*G+0.0722*B
        if(chanel == 3):
                Avg = 0.2627*R+0.6780*G+0.0593*B

        for i in range(3):
           new_img[:,:,i] = Avg
        return new_img

def Filter_(im, chanel):
    img = im[:,:,1]
    img_filter = img
    im_new = im
    w = 1

    high = len(img)
    width = len(img[0])

    for i in range(1,high-1):
        for j in range(1,width-1):
                img[i][j] = set_filter(img[i-1:i+2,j-1:j+2],chanel)

    for i in range(3):
        im_new[:,:,i] = img_filter

    return im_new

def set_filter(metrix, chanel_):
    if chanel_ == 1:
        return np.mean(metrix);
    elif chanel_ == 2:
        temp_med = np.reshape(metrix,(1,9))
        #print(metrix)
        temp_med = temp_med[0]
        return np.median(temp_med)

def filter_gau_(img_, sigma):
    return_img = img_;
    img_g = img_[:,:,0]
    heigh_ = len(img_g)
    width_ = len(img_g[0])
    new_img = img_g
    sum = 0

    kernel = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            ic = i - (len(kernel)-1)/2
            jc = j - (len(kernel[0])-1)/2

            g = (1/(2*sigma*sigma*np.pi))*np.exp(-(ic*ic+jc*jc)/(2*sigma*sigma))
            sum += g
            kernel[i][j] = g

    for i in range(3):
        for j in range(3):
            kernel[i][j] /= sum;


    print("kernel   :  \n",kernel)
    print("sum      :  \n",sum)
    print("pi       :  \n",np.pi)


    for i in range(heigh_-2):
        for j in range(width_-2):
            new_img[i][j] = np.sum(img_g[i:i+3,j:j+3]*kernel)
    for i in range(3):
        return_img[:,:,i] = new_img
    return  return_img

def img2binary(img_2, set_line):
    new_img = img_2
    new_img[new_img > set_line] = 255
    new_img[new_img <= set_line] = 0
    return new_img

def blob_coloring(blob, ob):
    blob_cut = blob[:,:,0].copy()
    count_ = 0
    lst = []
    equ = []
    new_blob  = blob[:,:,0]
    height = len(new_blob)
    width = len(new_blob[0])

    for i in range(height):
        for j in range(width):

            if new_blob[i][j] == 255:
                new_blob[i][j] = 0
            else:
                new_blob[i][j] = 255

            if(i == 0 or  j == 0 or j == width-1 ):
                print("",end="")
            elif new_blob[i][j] == 255:

                temp = [new_blob[i-1][j-1],new_blob[i-1][j],new_blob[i-1][j+1],new_blob[i][j-1]]
                if max(temp) == 0:
                    count_ += 1
                    new_blob[i][j] = count_
                else:
                    temp = [x for x in temp if x !=0]
                    new_blob[i][j] = min(temp)
                    if len(np.unique(temp))>1:
                        lst.append(np.unique(temp))
    equ.append([0])
    for ll in range(len(lst)):
        check_same = 0
        for x in range(len(equ)):
            for i in lst[ll]:
                if i in equ[x]:
                    #print (i, 'is in a')
                    print("lst:",lst[ll],"  :  ",equ[x], end="")
                    equ[x] = np.union1d(equ[x],lst[ll])
                    print("        ",equ[x] )
                    check_same = 1
                    break
        if check_same == 0:
            equ.append(lst[ll])
            print("lst:",lst[ll],"  +  ",equ[x],"  =  ",equ[x+1])
            equ.append(lst[ll])
        print("")

    print("============ Show Picture ===============")
      #for i in range(len(equ)):
    #    print(equ[i])

    for i in range(height):
        for j in range(width):
            for x in range(len(equ)):
                if new_blob[i][j] in equ[x]:
                    new_blob[i][j] = min(equ[x])

            if new_blob[i][j] == 0:
                print("  ",end="")
                blob_cut[i][j] = 255

            else:
                print("%2d" %new_blob[i][j],end="")
                if new_blob[i][j] == ob:
                    blob_cut[i][j] = 0
                else:
                    blob_cut[i][j] = 255
        print("")

    re_img = blob.copy()

    for i in range(3):
        re_img[:,:,i] = blob_cut

    return  re_img

def crop_picture(img_for_crop):
    a = img_for_crop[:,:,1]

    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    for i in range(len(a)):
        if 0 in a[i]:
            print("V",i)
            x1 = i
            break

    for i in range(len(a)-1,-1,-1):
         if 0 in a[i]:
            print("^",i)
            x2 = i
            break

    for i in range(len(a[0])):
        if 0 in a[:,i]:
            print(">",i)
            x3 = i
            break

    for i in range(len(a[0])-1,-1,-1):
        if 0 in a[:,i]:
            print("<",i)
            x4 = i
            break

    high_crop = x2-x1+1
    with_crop = x4-x3+1
    crop_image = cv2.resize(img_for_crop, (with_crop, high_crop))

    print("shape_1",img_for_crop.shape)
    print("shape_2",crop_image.shape)

    print(high_crop)
    print(with_crop)

    for i in range(x1,x2+1,1):
        for j in range(x3,x4,1):
            crop_image[i-x1,j-x3] = a[i][j]
            print("%3d"%crop_image[i-x1,j-x3,1],end="")

        print(" ")

    return crop_image


x = 178                                                 #int(input("input : "))
blob_ = 71
img = cv2.imread('blob.jpg',1)
gray_img = rgb_to_gray(img,1)                           #gray_scal

#Other function
image_ = filter_gau_(gray_img,1)                        #Gaussian
#image_ = Filter_(gray_img,1)                           #avg
#image_ = Filter_(gray_img,2)                           #median

binary_img = img2binary(image_,x)                       #binary
blob_coloring_img = blob_coloring(binary_img, blob_)    #blob
pic_ = crop_picture(blob_coloring_img)

#cv2.imshow("gau",image_)
#cv2.imshow("avg",image_)
#cv2.imshow("median",image_)

#plt.hist(gray_img.ravel(),256,[0,256])              #plot
cv2.imshow("crop_picture",pic_)
cv2.imshow("blob",blob_coloring_img)                 #blob
cv2.imshow("gray_img",gray_img)                     #gray_scal
cv2.imshow("origi",img)                             #origi

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()