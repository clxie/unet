#!/bin/env python
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
import numpy as np
import os
import shutil
im_w=100
im_h=100
im_sz=im_w*im_h
im_c=3
#convert cv2 img to PIL Image
def cvImg2PilImg(cvimg):
    pimg = Image.frombytes('L',cvimg.shape,cvimg.tostring())
    return pimg

#convert PIL img to numpy array
def PilImg2arr(pimg):
    arr = np.array(pimg)
    return arr

#convert arr to cv2 img
def arr2cvimg(arr):
    pimg = Image.fromarray(arr) 
    pimg2 = pimg.convert('RGB')
    cvimg = cv2.cvtColor(np.asarray(pimg2),cv2.COLOR_BGR2GRAY)
    return cvimg

def cvimg2arr(cvimg):
    pimg = cvImg2PilImg(cvimg)
    arr = PilImg2arr(pimg)
    return arr

#input 1D numpy array and threhold
#output 1D numpy array
def imgthres1d(arr, thres, bval1, lval2):
    arr2 = np.array(arr)
    idx1 = (arr[:] >= thres)
    idx2 = (arr[:] < thres)
    arr2[idx1] = bval1
    arr2[idx2] = lval2

    return arr2

def imgthres2d(arr, thres, bval1, lval2):
    arr2 = np.array(arr)
    idx1 = (arr[:,:] >= thres)
    idx2 = (arr[:,:] < thres)
    arr2[idx1] = bval1
    arr2[idx2] = lval2

    return arr2

#input 2D img[row, col]
#output 1D img[src:src:canny]
def img_process(arr):
    #arr1, threshold img
    mean = np.mean(arr)
    arr_copy = np.array(arr)
    arr_th = imgthres2d(arr_copy, mean, 255, 0)

    #arr2, canny edge img
    cvimg = arr2cvimg(arr)
    cvimgcanny = cv2.Canny(cvimg,100,180)

    arr3 = []#np.random.random(im_w*im_h*im_c).reshape([28,28,2])
    arr3.append(np.reshape(arr_th, [-1])) # mean value threshold 
    arr3.append(np.reshape(cvimg2arr(cvimgcanny), [-1])) # canny edge 
    arr3.append(np.reshape(arr, [-1])) #source img

    arr_x3 = np.reshape(arr3,[-1])
    return arr_x3

#calculate two imgs different pixel percent
def img_prediction(arr1,arr2):
    cnt = 0
    for i in range(im_w*im_h):
        if arr1[i] == arr2[i]:
            cnt += 1
    prediction = cnt / (im_w*im_h)

    return prediction

#input img path
#output 1D numpy array img
ol = 10
def read_img_data(path, method=0):
    img = (Image.open(path)).convert('L')
    (w,h) = img.size[:2]
    if w != h:
        print('Image width != height')
        return
    if method == 0:
        if (im_w == w & im_h == h):
            return np.reshape(np.array(img),[-1])
        img_rs = img.resize((im_w,im_h),Image.BOX)
        return np.reshape(img_rs,[-1])
    else:
        img_arrs = []
        step = im_h - ol * 2
        if w % step == 0:
            r = h / step
            c = w / step
        else:
            r = h // step + 1
            c = w // step + 1
        for row in range(int(r)):
            for col in range(int(c)):
                if col == (c - 1):
                    x0 = w - im_w
                    x1 = w
                else:
                    x0 = col * step
                    x1 = x0 + im_w

                if row == (r - 1):
                    y0 = h - im_h
                    y1 = h
                else:
                    y0 = row * step
                    y1 = y0 + im_h

                roi_box = (x0,y0,x1,y1)
                roi_img = img.crop(roi_box)
                img_arr = np.array(roi_img)
                img_arr2 = np.reshape(img_arr, 100*100)
                img_arrs.append(img_arr2)
        return img_arrs



#input img path
#output 1D img label
def read_label_data(path):
    label = read_img_data(path)
    arr_label = imgthres1d(label, 128, 1.0, 0.0)
    #convert 2D img to 1D
    #arr_label2 = np.reshape(arr_label, [im_sz])

    return arr_label

def read_dir_imgs(path,method=0):
    filelist = os.listdir(path)
    img_data=[]

    if method == 0:
        for i in range(len(filelist)):
            img_path = path + '/' + filelist[i]
            img = read_img_data(img_path)
            img_data.append(img)
    else:
        for i in range(len(filelist)):
            img_path = path + '/' + filelist[i]
            imgs = read_img_data(img_path,method)
            #for i in range(len(imgs)):
            #    img_data.append(imgs[i])
            img_data.extend(imgs)

    return np.array(img_data)

def read_dir_labels(path):
    filelist = os.listdir(path)
    label_data=[]

    for i in range(len(filelist)):
        label_path = path + '/' + filelist[i]
        label = read_label_data(label_path)
        label_data.append(label)
    return np.array(label_data)

#input train image dir which contains two child-dirs, images && labels
#output (x, label) data
def read_img_label_data(dirs):
    img_path = dirs + '/imgs'
    label_path = dirs + '/label'

    x = read_dir_imgs(img_path)
    label = read_dir_labels(label_path)

    return (x,label)

def get_batch_data(x,label,batch_size,step):
    (h,w) = x.shape[:2]
    (h2,w2) = label.shape[:2]

    x1 = np.zeros(batch_size * w).reshape([batch_size, w])
    label1 = np.zeros(batch_size * w2).reshape([batch_size, w2])

    start = step * batch_size
    end = (step + 1) * batch_size

    if batch_size > h:
        x1 = x
        label1 = label
    else:
        if ((start >= h) & (end >= h)): 
            start1 = start % h
            end1 = end % h
            b1 = end//h
            if start1 < end1:               # 1. the same window ---h---start---end---2h
                x1 = x[start1:end1,:]
                label1 = label[start1:end1,:]
            else:                           # 2. diff window  ---h---start---2h---end----
                m = h*b1 - start
                x1[0:m,:] = x[start1:h,:]
                x1[m:batch_size,:] = x[0:end1,:]

                label1[0:m,:] = label[start1:h,:]
                label1[m:batch_size,:] = label[0:end1,:]
        elif ((start < h) & (end >= h)):        # 3. ---start---h---end---
            m = h - start
            end1 = end % h

            x1[0:m,:] = x[start:h,:]
            x1[m:batch_size,:] = x[0:end1,:]

            label1[0:m,:] = label[start:h,:]
            label1[m:batch_size,:] = label[0:end1,:]
        else:                               # 4. ---start---end---h---
            x1 = x[start:end,:]
            label1 = label[start:end,:]
            #print('x shape (h=%g,w=%g), label shape (h=%g, w=%g) start=%g,end=%g' % (h,w,h2,w2,start,end))

    return (x1,label1)

def sobel_edge(imggray, outpath):
    imgsobel = cv2.Sobel(imggray, cv2.CV_16S, 1, 1, ksize=3)
    cv2.convertScaleAbs(imgsobel,imggray)
    cv2.imwrite(outpath, imggray)

def laplacian_edge(imggray, outpath):
    img_lap = cv2.Laplacian(imggray, cv2.CV_16S, ksize=3)
    cv2.convertScaleAbs(img_lap, imggray)
    cv2.imwrite(outpath, imggray)

def canny_edge(imggray, outpath):
    img_canny = cv2.Canny(imggray,50,100)
    cv2.imwrite(outpath, img_canny)

def build_canny_edge(inputdir, outputdir, method='canny'):
    if os.path.exists(inputdir):
        outputdir = outputdir + '_' + method
        if os.path.exists(outputdir):
            shutil.rmtree(outputdir)
        os.mkdir(outputdir)
        filelist = os.listdir(inputdir)
        print("############ Total images: %g" % (len(filelist)))
        for i in range(len(filelist)):
            img_path = inputdir + '/' + filelist[i];
            out_path = outputdir + '/' + filelist[i];
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if method == 'canny':
                canny_edge(img_gray,out_path)
            elif method == 'laplacian':
                laplacian_edge(img_gray,out_path)
            else:
                sobel_edge(img_gray,out_path)
    else:
        print("Input dir %s not exists!" % (inputdir))

def show_result(path,src):
    filelist = os.listdir(path)
    print("############ Total show images: %g" % (len(filelist)))
    plt.ion()
    for i in range(len(filelist)):
        print('%s --------> %g' % (filelist[i], i))
        img_path = path + '/' + filelist[i]
        src_path = src + '/' + filelist[i]
        img = Image.open(img_path)
        img2 = Image.open(src_path)
        plt.figure('Image')
        plt.subplot(121),plt.imshow(img),plt.title('src')
        plt.subplot(122),plt.imshow(img2),plt.title('contour')
        plt.pause(2)
        plt.close()
