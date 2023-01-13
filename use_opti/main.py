# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:46:46 2021

@author: admin
"""

import numpy as np
from os import listdir
from math import exp
from PIL import Image
import cv2
from dicom2png import *
from Hessian_egnv_optimize import *
from Judgle_flap import *
#-----------------------------------------------将人的一组DICOM文件转化为png文件-----------------------------------------------
import os
import SimpleITK as sitk
import dicom

import time

from predict import *

from debug_tool import *

import pandas as pd 

def is_dicom_file(filename):
    '''
       判断某文件是否是dicom格式的文件
    :param filename: dicom文件的路径
    :return:
    '''
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def load_patient(src_dir):
    '''
    读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = dicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()  
    reader.LoadPrivateTagsOn()
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    # print('series_id=',series_id)
    
    img_num = 0
    for i in range (len(series_id)):
        #print(i)
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id[i])
        #print(dicom_names)
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(dicom_names)
        
        try:
            image = series_reader.Execute()
        except:
            print("Error: 文件读取失败")
        else:
            img_array = sitk.GetArrayFromImage(image)
            img_array[img_array == -2000] = 0
        
            if img_array.shape[0] > img_num:
                if img_array.shape[1] == 512 and img_array.shape[2] == 512:
                    need_array = img_array
                    img_num = img_array.shape[0]
        
    return need_array

def normalize_hu(image):
    '''
           将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
       :param image 输入的图像数组
       :return: 归一化处理后的图像数组
    '''
    Src_img = image
    
    MIN_BOUND2 = -150.0
    MAX_BOUND2 = 350.0
    
    MIN_BOUND = -240.0
    MAX_BOUND = 260.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    
    image2 = (Src_img - MIN_BOUND2) / (MAX_BOUND2 - MIN_BOUND2)
    return image,image2
#==============================================求解图像像素点的海森矩阵特征值、经Frangi滤波器、黑白二值化=====================================
def Hessian_Frangi_Binary(img_use):
    [h,l] = img_use.shape
    flap_out = np.array( [ [0.0] * l for i in range(h) ] )                 #初始化Frangi滤波的结果
    flap_enhance = np.array( [ [0.0] * l for i in range(h) ] )
    eig_mat_large , eig_mat_small = Hessian_eigenvalue_ave(img_use,xx_ker_use,yy_ker_use,xy_ker_use,n)    #在此函数中调用 Hessian_scale_one
    binary_img = img_use
    binary_img[binary_img>0] = 255
    inf_list = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    line = inf_list[1]
    num = len(line)
    del_index = np.array([-1])
    #print( '原line数：',len(line) )
    for i in range(0 , len(line)):
        if len(line[i]) <20:
            del_index = np.append(del_index,i)
        else:
            area = cv2.contourArea(line[i])
            if area < 300 :
                del_index = np.append(del_index,i)
            #print('del_index：',del_index)
    for i in range(1 , len(del_index)):
        j= len(del_index)-i
        del line[del_index[j]]
    num = len(line)
                
    h = 512
    l = 512
                                
    if num > 0:
        temp_array = np.asarray(line[0])
        temp_array = temp_array.ravel()
        temp_array = temp_array.reshape(-1,2)
        line_array = [ temp_array ]
        #print(temp_array.shape)
        for index in range(1 , num):
            temp_array = np.asarray(line[index])
            temp_array = temp_array.ravel()
            temp_array = temp_array.reshape(-1,2)
            line_array.append(temp_array)
            
        for index in range(num):
            area = [np.min(line_array[index],axis = 0),np.max(line_array[index],axis = 0)]
            # area =[[minX,minY],[maxX,maxY]]
            for i in range(area[0][1],area[1][1] + 1):
                temp_eig_large = eig_mat_large[i]
                temp_eig_small = eig_mat_small[i]
                for j in range(l):
                    eig_large = temp_eig_large[j]
                    if eig_large > 0:
                        eig_small = temp_eig_small[j]
                        Rf = ( abs(eig_small)/abs(eig_large) )
                        Rn = ( eig_small * eig_small + eig_large * eig_large)**0.5
                        flap_out[i][j] = exp( -1*(Rf**2)/(2*a**2) ) * ( 1 - exp( (-1*Rn**2)/(2*b**2) ) )
                
    flap_enhance = flap_out * 255
    threshold_binary , flap_binary = cv2.threshold( flap_enhance , 150 , 255 , cv2.THRESH_BINARY )   #返回的第一个值为阈值，第二个是图像
    # flap_binary = cv2.adaptiveThreshold(flap_enhance.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)  #高斯处理
    return flap_binary 
#=================================================================图像开运算=========================================================
def smooth_point(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    open_img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations=1)
    
    return open_img

def log_diameters(diameters_list, patient_name):
    df = pd.DataFrame(diameters_list)
    if not os.path.isdir('diameters'):
        os.mkdir('diameters')
    df.to_excel('diameters'+ '/'+ patient_name + '.xlsx', index=False)
    
if __name__ == '__main__':
  dir_list = ['III']   ##testing dir
  for dir_index in range(len(dir_list)):
    dir_name = dir_list[dir_index]
    root_dir = 'D:/CTA/SUM/'+dir_name+'/'  
    train_file = open('D:/资料/work/主动脉夹层/U-net测试/Pytorch-UNet-master/train.txt','r')  ##读取参与了训练的数据名
    train_set = train_file.readlines()
    train_file.close()

    '''最终结果存储'''
    # path_txt_classify = root_dir + '0328_classify.txt'
    # record_classify_result = open ( path_txt_classify,'w' )
    
    total_time = 0
    total_cnt = 0
    
    # 导入神经网络训练的模型参数
    Model_path = 'new_model_dark.pth'  
    Model_flip_path = 'with_noise.pth'
    net = initial(Model_path)
    #net_xz = initial_xz(Model_xz_path)
    net_flip = initial_flip(Model_flip_path)
      
    for filename in listdir(root_dir):
      if not filename.startswith('.') and not filename[-4:] in '.txt' and not filename[-5:] in '.xlsx':
          
        ##是训练样本则读取下一个
        if ((filename+'\n') in train_set): 
            continue  
        filename = filename.split('/')[0]
        # filename = 'WANG DE FU_1'
        print('filename:',filename)
        path_now = os.path.join(root_dir , filename )
        image = get_pixels_hu_by_simpleitk(path_now)           
        slc_sum = len(image)
        slc_start , slc_end = int(0.4*slc_sum) ,  int(0.9*slc_sum)                         
        
        a,b = 0.8,0.4
        w = [5,7,9]                                                  #卷积高斯矩阵窗口大小                                    
        n = len(w)
        one_s = np.ones(n,float)
        w_2=[q*2 for q in w]
        sigema = (w_2 + one_s)/6
        xx_ker_use , yy_ker_use , xy_ker_use = Kernel_cal(w,n,sigema)
    
        ascend_list = []
        descend_list = []
        num_list = []
        slc_error_list = []
        ascend_temp_list = []
        descend_temp_list = []
        
        ascend_threshold  =  10                                         #升主动脉患病得分阈值
        descend_threshold = 10                                          #降主动脉患病得分阈值
    
        if not os.path.isdir(path_now + '/origin/'):
            mkdir(path_now + '/origin/')
        if not os.path.isdir(path_now + '/mask_result/'):
            mkdir(path_now + '/mask_result/')
        if not os.path.isdir(path_now + '/no_xz_result/'):
            mkdir(path_now + '/no_xz_result/')
        if not os.path.isdir(path_now + '/hessian_imgs2/'):
            mkdir(path_now + '/hessian_imgs2/')
        if not os.path.isdir(path_now + '/open_imgs/'):
            mkdir(path_now + '/open_imgs/')
        if not os.path.isdir(path_now + '/cut_smooth_imgs/'):
            mkdir(path_now + '/cut_smooth_imgs/')
        # mkdir(path_now + '/mask_ero_hessian/')
        
        slc_reverse = -520
        
        #建立状态流动站
        inn_length = 16
        status_num_flow = [0] * inn_length
        status_circle_flow = [0] * inn_length
        
        #图层所处阶段
        section = 0
        start = time.clock()
        
        ascend_sum = 0
        descend_sum = 0
        diameters_list = []
        for slc in range(slc_start,slc_end):
            slc_reverse = slc_end + slc_start - slc                     # 数组索引下标倒序访问
            # print( "slc_now:",slc_reverse )
            nor_image = normalize( image[slc_reverse] )
            nor_image=Image.fromarray((nor_image * 255).astype(np.uint8))
            
            # nor_image.save (path_now +'/origin/' +str(slc_reverse) + '.png')   #在每个人的文件夹下存储原始图像￥￥￥
            
            img_array , mask_array = cut_and_smooth( nor_image , net , slc_reverse , root_dir , filename)  #调试用slc_reverse,root_dir,filename保存信息￥￥￥
            img_save = Image.fromarray(img_array.astype(np.uint8))
            # img_save.save(path_now +'/cut_smooth_imgs/' +str(slc_reverse) + '.png')
            
            ascend_score , descend_score , num , circle_flag, diameters = Calculate_score( mask_array, img_array, \
                                                                                                  net_flip ,path_now + '/hessian_imgs2/', slc_reverse, filename)
            descend_sum += descend_score
            ascend_sum += ascend_score
            if not diameters.all() == 0: 
                diameters_list.append(diameters)
            
            if num >= 3 :
                num = -1                         #标记异常图片
                
            #更新状态栈
            status_num_flow.pop(0)                      # 第一个 num 状态出栈
            status_num_flow.append(num)                 # 尾添 最新图层 num
            status_circle_flow.pop(0)                   # 第一个 circle_flag 出栈
            status_circle_flow.append(circle_flag)      # 尾填 最新图层 circle_flag
            
        if ascend_sum > ascend_threshold and descend_sum > descend_threshold :
            print(str(filename),'Type I')
        elif ascend_sum > ascend_threshold and descend_sum <=descend_threshold:
            print(str(filename),'Type II')
        elif ascend_sum <= ascend_threshold and descend_sum > descend_threshold:
            print(str(filename),'Type III')
        elif ascend_sum <= ascend_threshold and descend_sum <= descend_threshold :
            print(str(filename),'Normal')
            break
        
        end = time.clock()
        total_time += end-start
        total_cnt += 1
        print('total_time:', total_time)
        print('total_cnt:', total_cnt)
        log_diameters(diameters_list, filename)
        
#========================================================================================================================================        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
