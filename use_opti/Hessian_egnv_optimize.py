# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 09:47:07 2021

@author: admin
"""

import numpy as np
import time
from os import listdir
import math
# import matrix as mat
import cv2
from PIL import Image


#需输入黑白图像的矩阵形式，k为白色连通域的数量，k=1时，仅有line_array[0]；k=2时，有line_array[0/1]。line_array为白色连通域的轮廓列表，其元素为边缘行列值[lie,hang]
'''
def Judge_white(img_array,k,line_array,one_circle_threshold,threshold_0,threshold_1):     
     [h,l] = img_array.shape
     if k==1 :                                                  #单区域            注意 line_array中的元素[lie，hang]
        white_list=[]
        find_flag = 1
        for i in range(h):
            for j in range(l):
                 if [j,i] in line_array[0] and [j+1,i] not in line_array[0]:   #确保从内边界开始搜索白块数量
                    find_flag = 1
                    a=1
                    while [j+a,i] not in line_array[0]  and j+a < l :
                        if img_array[i][j+a]==255 :
                            white_list.append([i,j+a])
                        else :
                            pass
                        a += 1
                    break
            if find_flag == 1 and j == l-1 :
                break
        record_news.write( '单圆白色块数量：' + str(len(white_list))+'\n' )
        if len(white_list) >= one_circle_threshold :
            record_news.write('单个圆白色数量异常\n')
     elif k==2 :                                               #双区域            注意line_array[0/1]中的元素为[lie,hang]
         white_list_0 = []
         white_list_1 = []
         white_list = []
         for i in range(h) :
             for j in range(l) :
                 # print( [j,i] in line_array[0] and [j+1,i] not in line_array[0] )
                 if ([j,i] in line_array[0]) and ([j+1,i] not in line_array[0]):   #确保从内边界开始搜索白块数量
                     a=1
                     # print('s',line_array[0])
                     while ([j+a,i] not in line_array[0]) and (j+a < l) :
                         # print('0区域边缘点：',[j+a,i],img_array[i][j+a])
                         if img_array[i][j+a]==255:
                             white_list_0.append([i,j+a])
                         else :
                            pass
                         a += 1
                     break
                 
                 if ([j,i] in line_array[1]) and ([j+1,i] not in line_array[1]):   #确保从内边界开始搜索白块数量
                     a=1
                     while ([j+a,i] not in line_array[1]) and (j+a < l) :
                         if img_array[i][j+a]==255:
                             white_list_1.append([i,j+a])
                         else :
                             pass
                         a += 1
                     break
         record_news.write('双圆区域0白色块数量：' + str(len(white_list_0)) + '\n')
         if  len(white_list_0) >= threshold_0:
                record_news.write('双圆0区域异常\n')
         record_news.write('双圆区域1白色块数量：' + str(len(white_list_1)) + '\n')
         if  len(white_list_1) >= threshold_1:
                record_news.write('双圆1区域异常\n')
         # image_to_excel_xyf(line_array[0])
'''        
def get_egien(img,xx_grads,yy_grads,xy_grads):
    binary_img = img
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
    
    eigen_mat_large = np.array( [ [0.0] * l for i in range(h) ] )                 #创建输出矩阵（处理后的图像）
    eigen_mat_small = np.array( [ [0.0] * l for i in range(h) ] )
    
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
                for j in range(area[0][0],area[1][0] + 1):
                    if img[i][j] > 0:
                        fxx = xx_grads[i][j]
                        fyy = yy_grads[i][j]
                        fxy = xy_grads[i][j]
                        if ( abs(fxx)+abs(fyy) ) != 0 :
                            hess_array = np.array( [ [fxx,fxy],[fxy,fyy] ] )
                            eigenvalue,vector = np.linalg.eig(hess_array)
                            eig0 = eigenvalue[0]
                            eig1 = eigenvalue[1]
                            if abs(eig0)<abs(eig1) :                 #保证lamuda_two是特征值绝对值较大的那个 
                                lamuda_one=eig0
                                lamuda_two=eig1      
                            else :
                                lamuda_one=eig1
                                lamuda_two=eig0
                            
                            eigen_mat_large[i][j] = lamuda_two
                            eigen_mat_small[i][j] = lamuda_one
    return eigen_mat_large,eigen_mat_small

def Kernel_cal(w,n,sigema):  
    PI=math.pi
    xx_ker_list =[]
    yy_ker_list =[]
    xy_ker_list =[]
    for a in range(n):
       xx_kernel = np.array( [ [0.0] * (2*w[a]+1) for i in range((2*w[a]+1))] )  
       yy_kernel = np.array( [ [0.0] * (2*w[a]+1) for i in range((2*w[a]+1))] )
       xy_kernel = np.array( [ [0.0] * (2*w[a]+1) for i in range((2*w[a]+1))] )
       for i in range(-w[a],w[a]+1):
         for j in range(-w[a],w[a]+1):
             xx_kernel[i+w[a]][j+w[a]]=float(( 1 - (i*i) / (sigema[a]*sigema[a]))* math.exp( -1 * (i*i + j*j)*1.0 / ( 2 * sigema[a]*sigema[a]))*( -1 / ( 2 * PI* math.pow(sigema[a], 4)) ))
             yy_kernel[i+w[a]][j+w[a]]=float(( 1 - (j*j) / (sigema[a]*sigema[a]))* math.exp( -1 * (i*i + j*j)*1.0 / ( 2 * sigema[a]*sigema[a]))*( -1 / ( 2 * PI* math.pow(sigema[a], 4)) ))
             xy_kernel[i+w[a]][j+w[a]]=float(( (i*j))* math.exp( -1 * (i*i + j*j)*1.0 / ( 2 * sigema[a]*sigema[a]))*( 1 / ( 2 * PI* math.pow(sigema[a], 6)) ))
       xx_ker_list.append(xx_kernel)      
       yy_ker_list.append(yy_kernel)
       xy_ker_list.append(xy_kernel)
    return xx_ker_list , yy_ker_list , xy_ker_list

def Hessian_scale_one( img_in  ,xx_kernel , yy_kernel , xy_kernel ):
    [h,l] = img_in.shape
    eigen_mat_large = np.array( [ [0.0] * l for i in range(h) ] )                 #创建输出矩阵（处理后的图像）
    eigen_mat_small = np.array( [ [0.0] * l for i in range(h) ] )
    # xx_kernel,yy_kernel,xy_kernel=Kernel_cal(w,sigema)                          #调用定义好的卷积核结果，不可重复计算
    xx_grads=cv2.filter2D( img_in,-1,xx_kernel )              #卷积高斯二阶导数矩阵，求取二阶偏导数
    yy_grads=cv2.filter2D( img_in,-1,yy_kernel )
    xy_grads=cv2.filter2D( img_in,-1,xy_kernel )
    
    eigen_mat_large, eigen_mat_small = get_egien(img_in,xx_grads,yy_grads,xy_grads)
            
    return eigen_mat_large,eigen_mat_small
                
def Hessian_eigenvalue_ave(img_in,xx_ker_use,yy_ker_use,xy_ker_use,n):
    [h,l]=img_in.shape
    eigenvalue_large = np.zeros( (h,l,n),float )
    eigenvalue_small = np.zeros( (h,l,n),float )
    eigenvalue_large_sum   = np.zeros( (h,l),float )
    eigenvalue_small_sum   = np.zeros( (h,l),float )
    k_large=[6,4,2]
    k_small=[1,3,5]
    sum_k_large = 0
    sum_k_small = 0
    
    for a in range(n):
        
        eigenvalue_large[:,:,a] , eigenvalue_small[:,:,a] = Hessian_scale_one( img_in  ,xx_ker_use[a] , yy_ker_use[a] , xy_ker_use[a])
    # for m in range(n):
        eigenvalue_large_sum += eigenvalue_large[:,:,a] * k_large[a]             #求和
        eigenvalue_small_sum += eigenvalue_small[:,:,a] * k_small[a]
        sum_k_large += k_large[a]
        sum_k_small += k_small[a]
    eigenvalue_large_ave = eigenvalue_large_sum/sum_k_large                   #求绝对值较大特征值的平均值
    eigenvalue_small_ave = eigenvalue_small_sum/sum_k_small                    #求绝对值较小特征值的平均值
    return eigenvalue_large_ave , eigenvalue_small_ave
    
# def main():
#     a=0.8
#     b=0.4
#     in_dir= 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\for_judge3\\AR_denoise\\'
#     w = [5,7,9]                                                  #卷积高斯矩阵窗口大小                                    
#     n = len(w)
#     one_s = np.ones(n,float)
#     w_2=[q*2 for q in w]
#     sigema = (w_2 + one_s)/6
#     xx_ker_use , yy_ker_use , xy_ker_use = Kernel_cal(w,n,sigema)
    
#     for dress_seq_c in listdir(in_dir) :
#         start =time.clock()
#         if not dress_seq_c.startswith('.') :
#             img_use = np.array( Image.open( in_dir + dress_seq_c ) )        #读取图像
#             [h,l] = img_use.shape
#             flap_out = np.array( [ [0.0] * l for i in range(h) ] )                 #初始化Frangi滤波的结果
#             flap_enhance = np.array( [ [0.0] * l for i in range(h) ] )
            
#             eig_mat_large , eig_mat_small = Hessian_eigenvalue_ave(img_use,xx_ker_use,yy_ker_use,xy_ker_use,n)    #在此函数中调用 Hessian_scale_one
#             for i in range(h):
#                 for j in range(l):
#                     if img_use[i][j] != 0:
#                         if eig_mat_large[i][j] <= 0 :
#                             flap_out[i][j] = 0
#                         else :
#                             Rf = ( abs(eig_mat_small[i][j])/abs(eig_mat_large[i][j]) )
#                             Rn = ( eig_mat_small[i][j]**2 + eig_mat_large[i][j]**2 )**0.5
#                             flap_out[i][j] = math.exp( -1*(Rf**2)/(2*a**2) ) * ( 1 - math.exp( (-1*Rn**2)/(2*b**2) ) )
#                     else: 
#                         flap_out[i][j] = 0
#             flap_enhance = flap_out * 255
#             img_save = Image.fromarray(flap_enhance.astype(np.uint8))
#             img_save.save('D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\for_judge3\\hessian_denoise2\\'+dress_seq_c)
#             print( str(dress_seq_c) )
#             end = time.clock()
#             print('Running time: %s Seconds'%(end-start))
    
#     '''测试单张'''
    # path = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\normal\\AR_denoise\\'
    # dress_seq_c = '0320_I1.png'
    # img_use = np.array( Image.open( path + dress_seq_c ) )        #读取图像
    # [h,l] = img_use.shape
    # flap_out = np.array( [ [0.0] * l for i in range(h) ] )                 #初始化Frangi滤波的结果
    # flap_enhance = np.array( [ [0.0] * l for i in range(h) ] )
    # w = [6,7,8,9,10,11]                                                  #卷积高斯矩阵窗口大小                                    
    # n = len(w)
            
    # eig_mat_large , eig_mat_small = Hessian_eigenvalue_ave(img_use,w,n)    #在此函数中调用 Hessian_scale_one
    # for i in range(h):
    #     for j in range(l):
    #         if img_use[i][j] != 0:
    #             if eig_mat_large[i][j] <= 0 :
    #                 flap_out[i][j] = 0
    #             else :
    #                 Rf = ( abs(eig_mat_small[i][j])/abs(eig_mat_large[i][j]) )
    #                 Rn = ( eig_mat_small[i][j]**2 + eig_mat_large[i][j]**2 )**0.5
    #                 flap_out[i][j] = math.exp( -1*(Rf**2)/(2*a**2) ) * ( 1 - math.exp( (-1*Rn**2)/(2*b**2) ) )
    #         else: 
    #             flap_out[i][j] = 0
    # flap_enhance = flap_out * 255
    # img_save = Image.fromarray(flap_enhance.astype(np.uint8))
    # img_save.save('D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\normal\\'+dress_seq_c)
    # print( str(dress_seq_c) )
            
# main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    