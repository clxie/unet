#!/bin/env python

import cv2
from PIL import Image
import numpy as np
import os
import read_data
import shutil

im_w = 100
im_h = 100

#convert cv2 img to PIL Image
def cvImg2PilImg(img):
    im = Image.frombytes('L',img.shape,img.tostring())
    return im

#convert PIL img to numpy array
def PilImg2arr(im):
    arr = np.array(im)
    return arr

#convert arr to cv2 img
def arr2cvimg(arr):
    pimg = Image.fromarray(arr) 
    pimg2 = pimg.convert('RGB')
    img = cv2.cvtColor(np.asarray(pimg2),cv2.COLOR_BGR2GRAY)
    return img

def cvimg2arr(img):
    im = cvImg2PilImg(img)
    arr = PilImg2arr(im)
    return arr

def rotate(img, name, outdir):
    (h,w) = img.shape[:2]
    center = (w/2,h/2)
    angle90 = 90 
    angle180 =  180
    angle270 = 270 
    scale = 1.0
    
    #save source img
    name0 = outdir + '/' + name
    cv2.imwrite(name0, img)

    pos = name.find('.tif')
    #90 degrees
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    img90 = cv2.warpAffine(img, M, (h,w))
    name90 = outdir + '/' + name[:pos] + '_' + str(90) + name[pos:]
    cv2.imwrite(name90, img90)

    #180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    img180 = cv2.warpAffine(img, M, (h,w))
    name180 = outdir + '/' + name[:pos] + '_' + str(180) + name[pos:]
    cv2.imwrite(name180, img180)

    #270 degrees
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    img270 = cv2.warpAffine(img, M, (h,w))
    name270 = outdir + '/' + name[:pos] + '_' + str(270) + name[pos:]
    cv2.imwrite(name270, img270)

def imgthres(arr, thres, bval1=255, lval2=0):
    arr2 = np.array(arr)
    idx1 = (arr2[:,:] >= thres)
    idx2 = (arr2[:,:] < thres)
    arr2[idx1] = bval1 
    arr2[idx2] = lval2

    return arr2


#input PIL Image, image name, output dir
#output im_w*im_h imgs to output dir
def clippimg(img, clip_sz, name, outdir):
    arr = np.array(img)
    (h,w) = arr.shape[:2]
    w_cnt = w//clip_sz
    h_cnt = h//clip_sz

    #save clip imgs
    for r in range(h_cnt):
        for c in range(w_cnt):
            arr1 = arr[r*clip_sz:(r+1)*clip_sz, c*clip_sz:(c+1)*clip_sz]
            pos = name.find('.')
            nn = outdir + '/' + name[:pos] + '_' + str(clip_sz) + '_' + str(r) + '_' + str(c) + name[pos:]
            pimg = Image.fromarray(arr1)
            pimg.save(nn)

def clippimg2(img, tgsz, name, outdir, step=50):
    arr = np.array(img)
    (h,w) = arr.shape[:2]

    all_inter = w - tgsz
    if all_inter % step == 0:
        cnt = all_inter / step
    else:
        cnt = (all_inter // step) + 1

    #save clip imgs
    for r in range(cnt):
        for c in range(cnt):
            if c == cnt - 1:
                x0 = w - tgsz
                x1 = w
            else:
                x0 = c * step
                x1 = c * step + tgsz

            if r == cnt - 1:
                y0 = h - tgsz
                y1 = h
            else:
                y0 = r * step
                y1 = r * step + tgsz

            arr1 = arr[y0:y1, x0:x1]
            pos = name.find('.')
            nn = outdir + '/' + name[:pos] + '_' + str(tgsz) + '_' + str(r) + '_' + str(c) + name[pos:]
            pimg = Image.fromarray(arr1)
            pimg.save(nn)

def clipimgs(img, name, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #save source img to outdir
    n0 = outdir + '/' + name
    img.save(n0)
    (w,h) = img.size[:2]
    if w != h:
        print('Image %s width != height' % (name))
        return
    cnt = w // 100

    for c in range(1, cnt):
        cliped_size = c * 100
        clippimg(img, cliped_size, name, outdir)

def clipimgs2(img, name, outdir, tgsz, step):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    #save source img to outdir
    n0 = outdir + '/' + name
    img.save(n0)
    (w,h) = img.size[:2]
    if w != h:
        print('Image %s width != height' % (name))
        return
    if w % tgsz == 0:
        cnt = w / tgsz
    else:
        cnt = (w // tgsz) + 1

    clippimg2(img, tgsz, name, outdir, step)

def rotatedirimgs(path, outdir):
    if os.path.exists(path):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        files = os.listdir(path)
        for i in range(len(files)):
            if files[i].find('.tif') > 0:
                imgp = path +'/' + files[i]
                cvimg = cv2.imread(imgp, 0)
                rotate(cvimg, files[i], outdir)
    else:
        print('Input img path: %s not exist!' % (path))

#rotatedirimgs(path=path,outdir=outdir)
#rotatedirimgs(path=path2,outdir=outdir2)

#clippimg(pimg, na, 'testtmpdir')
#clipimgs(pimg, na, 'testtmpdir2')

def clipdirimgs(path, outdir):
    if os.path.exists(path):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        files = os.listdir(path)
        for i in range(len(files)):
            if (files[i].find('.tif') > 0):
                imgp = path + '/' + files[i]
                pimg = Image.open(imgp).convert('L')
                clipimgs(pimg, files[i], outdir)

def clipdirimgs2(path, outdir):
    if os.path.exists(path):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        files = os.listdir(path)
        for i in range(len(files)):
            if (files[i].find('.tif') > 0):
                imgp = path + '/' + files[i]
                pimg = Image.open(imgp).convert('L')
                clipimgs2(pimg, files[i], outdir, 256, 50)

def clipim(path, outdir):
    #rotate img
    tmpdir = 'tmp_img_rotates'
    rotatedirimgs(path, tmpdir)

    #clip rotated imgs
    clipdirimgs2(tmpdir, outdir)

    shutil.rmtree(tmpdir)

path = '/mnt/pc-hf197/chunleixie/ml/mnist/all/train/imgssrc/'
outd = '/mnt/pc-hf197/chunleixie/ml/mnist/all/train/imgs/'

path2 = '/mnt/pc-hf197/chunleixie/ml/mnist/all/train/labelsrc/'
outd2 = '/mnt/pc-hf197/chunleixie/ml/mnist/all/train/label/'

### rotate and clip images to 100*100 
print('in:%s,out:%s' % (path,outd))
clipim(path,outd)
print('in:%s,out:%s' % (path2,outd2))
clipim(path2,outd2)

### canny laplican sobel edge 
path1 = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/imgssrc116/'
path2 = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/imgssrc116_rotate'
path3 = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/label_laplacian/'
path4 = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/labeledge_rotate/'

#rotatedirimgs(path1, path2)
#rotatedirimgs(path3, path4)
#outd = '/mnt/pc-hf198/chunleixie/ml/mnist/all/train/label'
#read_data.build_canny_edge(path,outd,'canny')
#read_data.build_canny_edge(path,outd,'laplacian')
#read_data.build_canny_edge(path,outd,'sobel')

#convert gray image to RGB
def convertImgToRGB(inputdir,outputdir):
    if os.path.exists(inputdir):
        if os.path.exists(outputdir):
            shutil.rmtree(outputdir)
        os.mkdir(outputdir)

        filelist = os.listdir(inputdir)
        print("########### Total imgs: %g ############" % (len(filelist)))
        for i in range(len(filelist)):
            img_path = inputdir + '/' + filelist[i]
            out_path = outputdir + '/' + filelist[i]
            img = cv2.imread(img_path)
            #print('img shape:',img.shape)
            cv2.imwrite(out_path, img)
    else:
        print("Input dir %s not exists!" % (inputdir))

#path1 = '/mnt/pc-hf198/chunleixie/ml/Keras_HED-master/dataset/HED-BSDS/train2/imgs2'
#path1_out = '/mnt/pc-hf198/chunleixie/ml/Keras_HED-master/dataset/HED-BSDS/train2/imgs'
#path2 = '/mnt/pc-hf198/chunleixie/ml/Keras_HED-master/dataset/HED-BSDS/train2/label'
#path2_out = '/mnt/pc-hf198/chunleixie/ml/Keras_HED-master/dataset/HED-BSDS/train2/label_out'

#convertImgToRGB(path1, path1_out)
