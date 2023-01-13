# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:20:08 2021

@author: admin
"""

import numpy as np
import sys
import time
from os import listdir
import urllib
import json
import re
import math
import decimal
import logging
import torch
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter
import cv2
#from split import *

save_path0 = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\type\\0\\'
save_path1 = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\type\\1\\'
save_path2 = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\type\\2\\'
save_path3 = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\type\\3\\'
save_path4 = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\type\\4\\'

threhold = 150

def Judge_circle(img_in,line,record_news): 
    # inf_list = cv2.findContours(img_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # line = inf_list[0]
    s_area=cv2.contourArea(line[0])                                            #CV2库函数计算封闭曲线面积
    #list[0]是轮廓的列表集，list[0] [0]是列表集的第一个元素[[h,l]],故第三个[0]消列表括号，第四个[]则调用或者列
    #故调用方式为list[0][pin][0][h/l]    非常蠢，后面降维了
    pin_sum = len(line[0])
    line_array = np.asarray(line)
    line_array = line_array.ravel()                                            #数组降维,前面的调用方式太蠢了！！！
    line_array = line_array.reshape(-1,2)           #降维后重塑数组，此时调用形式为line_array[pin][lie=0/hang=1]   python牛皮！  注意这个轮廓[列x，行y]
    # print(line_array)
    center_h,center_l = Find_center_circle(line_array,pin_sum)
    R_sq_max=0
    for p in range(pin_sum):
        R_sq =( line_array[p][1]-center_h )**2+( line_array[p][0]-center_l )**2
        if R_sq > R_sq_max:
          R_sq_max = R_sq
    s_max = math.pi * R_sq_max
    ratio_s = s_max/s_area
    record_news.write( '单区域圆度参数  ' +str(s_max)+'   ' + str(s_area)+'   ' + str(ratio_s)+str('\n') )
    if ratio_s >=2 :
        if s_area>7000:
            record_news.write('长条形\n')
            return 2 , line_array
        else:
            record_news.write('圆形\n')
            return 1 , line_array
    else :
        record_news.write('圆形\n')
        return 1 , line_array
    # cv2.drawContours(img_in,line,0,(0,0,255),3)  
    # print(line[0][0][0][0],type(line),len(line[0]))
#---------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------寻找轮廓中心-------------------------------------------------------------------    
def Find_center_circle(line_array,pin_sum):
    h_pos = 0
    l_pos = 0
    for i in range(pin_sum):
        h_pos += line_array[i][1]
        l_pos += line_array[i][0]
    center_h = float(h_pos)/pin_sum
    center_l = float(l_pos)/pin_sum
    return center_h,center_l
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------求有几个连通域-----------------------------------------------------------------
class Stack(object):
    """栈"""
    def __init__(self):
         self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items == []

    def push(self, item):
        """加入元素"""
        self.items.append(item)

    def pop(self):
        """弹出元素"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈的大小"""
        return len(self.items)
def cnt_link_area(src_img):
    ''' src_img --- 需要计算的图片'''
    img_array = np.array(src_img)
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    img_array = np.pad(img_array, ((1, 1), (1, 1)), 'constant')
    
    label_img = np.ones((rows,cols))*0
    label_img = np.pad(label_img, ((1, 1), (1, 1)), 'constant')
    number =0
    
    '''遍历图片 '''
    for i in range(1 , rows):
        for j in range(1 , cols):
            '''寻找种子点'''
            cnt = 0
            if label_img[i][j] == 0 and img_array[i][j] == 255:
                area_point = Stack()   #入栈
                area_point.push([i,j])
                
                number += 1            #统计数量
                label_img[i][j] = number #标记
                cnt += 1
                
                '''从种子点开始寻找连通域'''
                while not area_point.is_empty():
                    cur_point = area_point.pop()
                    cur_i = cur_point[0]
                    cur_j = cur_point[1]
                    
                    #print("cur_i:",cur_i,"cur_j:",cur_i)
                    
                    for t in range(-1,2):
                        for p in range(-1,2):
                            if label_img[cur_i + t][cur_j + p] == 0 and img_array[cur_i + t][cur_j + p] == 255:
                                area_point.push([cur_i + t,cur_j + p])
                                label_img[cur_i + t][cur_j + p] = number
                                cnt += 1
                if cnt <= 30:
                    number -= 1
    return number
#----------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------求取轮廓最小矩形框并给出分割线--------------------------------------------------------  
def split(img_in,counter,center_x,center_y):  
    rect = cv2.minAreaRect(counter)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print(box)
    
    
    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
     
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # 上下左右四个点坐标
    vertices = np.array([[bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [top_point_x, top_point_y], [right_point_x, right_point_y]])
    box = vertices
    #for i in range (4):
    #    cv2.line(src_img,(box[i%4][0],box[i%4][1]),(box[(i+1)%4][0],box[(i+1)%4][1]),(255,255,255),2)
    
    length1 = (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2
    length2 = (box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2
    
    if length1 >= length2:  ##往左斜
        k_rec = float(box[1][1] - box[2][1])/(box[1][0] - box[2][0])
        b_up = box[1][1] - k_rec*box[1][0]
        b_down = box[3][1] - k_rec*box[3][0]
        
        k_mid = k_rec
        b_mid = center_y - k_mid * center_x
        
        up_area = np.array([[box[1][0],(box[2][0]+box[3][0])/2],[box[2][1],(box[0][1]+box[1][1])/2]]) #[minx,maxx],[miny,maxy]
        down_area = np.array([[(box[0][0]+box[1][0])/2,box[3][0]],[(box[2][1]+box[3][1])/2,box[0][1]]]) #[minx,maxx],[miny,maxy]
    
    else: ##往右斜
        k_rec = float(box[1][1] - box[0][1])/(box[1][0] - box[0][0])
        b_up = box[2][1] - k_rec*box[2][0]
        b_down = box[0][1] - k_rec*box[0][0]
        k_mid = k_rec
        b_mid = center_y - k_mid * center_x
        
        up_area = np.array([[(box[1][0]+box[2][0])/2,box[3][0]],[box[2][1],(box[0][1]+box[3][1])/2]]) #[minx,maxx],[miny,maxy]
        down_area = np.array([[box[1][0],(box[0][0]+box[3][0])/2],[(box[1][1]+box[2][1])/2,box[0][1]]]) #[minx,maxx],[miny,maxy]

    return k_mid,b_mid  
#----------------------------------------------------------------数白色像素点的个数--------------------------------------------------------------
#需输入黑白图像的矩阵形式，k为白色连通域的数量，k=1时，仅有line_array[0]；k=2时，有line_array[0/1]。line_array为白色连通域的轮廓列表，其元素为边缘行列值[lie,hang]
def Judge_white( img_array , k , line_array , center_h0, center_h1, one_circle_threshold , threshold_0 , threshold_1 ,record_news):     
     [h,l] = img_array.shape
     if k==1 :                                                  #单区域            注意 line_array中的元素[lie，hang]
        white_list=[]
        find_flag = 0
        for i in range(h):
            for j in range(l):
                 if [j,i] in line_array and [j+1,i] not in line_array:   #确保从内边界开始搜索白块数量
                    find_flag = 1
                    a=1
                    while [j+a,i] not in line_array  and j+a < l :
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
            return 5
        else:
            return 0
        
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
         if center_h0 <= center_h1:
             white_list_up = white_list_0
             threshold_up = threshold_0
             white_list_down = white_list_1
             threshold_down = threshold_1
         else:
             white_list_up = white_list_1
             threshold_up = threshold_1
             white_list_down = white_list_0
             threshold_down = threshold_0
            
         record_news.write('双圆区域up白色块数量：' + str(len(white_list_up)) + '\n')
         if  len(white_list_up) >= threshold_up:
                record_news.write('双圆up区域异常\n')
         record_news.write('双圆区域down白色块数量：' + str(len(white_list_down)) + '\n')
         if  len(white_list_down) >= threshold_down:
                record_news.write('双圆down区域异常\n')
                
         if len(white_list_up) >= threshold_up and len(white_list_down) >= threshold_down:
             return 6
         elif len(white_list_up) >= threshold_up:
             return 7
         elif len(white_list_down) >= threshold_down:
             return 8
         else:
             return 9
             
         
         # image_to_excel_xyf(line_array[0])   
#--------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------统计椭圆内两区域的白像素点个数-----------------------------------------------
def Judge_white_oval(img_array,line_array,k,b,threshold,record_news):
        [h,l] = img_array.shape
        white_list_up = []         #在分割线上
        white_list_down =[]        #在分割线下
        white_list_kb =[]          #正好在分割线上
        find_flag = 0
        for i in range(h):
            for j in range(l):
                 # print(line_array)
                 #print([j,i])
                 if [j,i] in line_array and [j+1,i] not in line_array:   #确保从内边界开始搜索白块数量
                    find_flag = 1
                    a=1
                    #print('区域数值',i-k*(j+a)-b)
                    while [j+a,i] not in line_array  and j+a < l :
                        if img_array[i][j+a]==255 :
                            if i-k*(j+a)-b < 0 :
                               white_list_up.append([i,j+a])
                            elif i-k*(j+a)-b > 0 :
                                white_list_down.append([i,j+a])
                            else :
                                white_list_kb.append([i,j+a])
                        else :
                            pass
                        a += 1
                    break
            if find_flag == 1 and j == l-1 :
                break
        print( len(white_list_up) , len(white_list_down) )
        record_news.write( '长条形白色块数量：  ' + str(len(white_list_up)) + '    '+ str(len(white_list_down)) +'\n')
        if len(white_list_up) >= threshold and len(white_list_down) < threshold :             #  1  仅上区域多白点
            return 1
        elif len(white_list_up) < threshold and len(white_list_down) >= threshold :           #  2  仅下区域多白点
            return 2
        elif len(white_list_up) >= threshold and len(white_list_down) >= threshold :          #  3  上下两区域都有很多白点
            return 3
        else :                                                                                #  4  上下两区域都没有白点
            return 4 
        
#---------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------对单连通域和双连通域进行分类处理-----------------------------------------------
def Classify_one_or_two(img_in,flap_in,name,record_news):
    inf_list = cv2.findContours(img_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    line = inf_list[0]                                                               #此处徐用的是inf_list[0],黄用的是inf_list[1]
    print(line[0])
    num = len(line)
    print(num)
    record_news.write( '连通区域'+str(num)+'\n' )
    flap_save = Image.fromarray(flap_in.astype(np.uint8))
    temp_img = np.ones((512,512))*0
    #cv2.drawContours(temp_img,line,1,(255,255,255),1)
    #cv2.drawContours(temp_img,line,0,(255,255,255),5)
    #cv2.imshow('temp_img',temp_img)
    #cv2.waitKey()
    
    line_num = np.zeros(1)
    #print(line)
    del_index = np.array([-1])
    for i in range(0 , len(line)):
        if len(line[i]) <20:
            del_index = np.append(del_index,i)
    for i in range(1,len(del_index)):
        del line[del_index[i]]
    num = len(line)
    
    if num == 1 :
        circle_flag , line_array = Judge_circle(img_in,line,record_news)      #对于单连通区域，判断类圆形的形状区分为长条形和类圆形,这一步将line降维到line_array
        line_list = line_array.tolist()                           #将line转为array形式,为方便使用in函数判断点的列行值[j,i]是否在序列上
        
        if circle_flag==1 :                                       #是圆状时：找白块
            result = Judge_white(flap_in,num,line_list, 0 , 0 ,150,150,150,record_news)
            if result == 0:
                flap_save.save(save_path0 + name)
            elif result == 5:
                flap_save.save(save_path1 + name)
                
        elif circle_flag==2 :
            center_oval_h , center_oval_l = Find_center_circle( line_array,len(line_array) )
            k , b = split(img_in,line[0],center_oval_l,center_oval_h)
            result = Judge_white_oval( flap_in ,line_list, k , b ,150,record_news)
            print( k,b )
            print( result )
            if result == 4:
                flap_save.save(save_path0 + name)
            elif result == 1:
                flap_save.save(save_path2 + name)
            elif result == 2:
                flap_save.save(save_path3 + name)
            elif result == 3:
                flap_save.save(save_path4 + name)
            
    elif num == 2 :
        line_array = [ np.asarray(line[0]),np.asarray(line[1]) ]
        line_array = [ line_array[0].ravel(),line_array[1].ravel() ]
        line_array = [ line_array[0].reshape(-1,2),line_array[1].reshape(-1,2) ]
        line_list=[line_array[0].tolist(),line_array[1].tolist()]
        #line_array[0]和line_array[1]分别为两个边缘轮廓的信息，使用方法line_array[0/1][pin][0-(lie)/1-(hang)]
        # print(line_array[0])
        num_list = [len(line_array[0]),len(line_array[1])]
        center_h_0,center_l_0 = Find_center_circle(line_array[0],num_list[0])  
        center_h_1,center_l_1 = Find_center_circle(line_array[1],num_list[1])
        result = Judge_white(flap_in,num,line_list,center_h_0,center_h_1,150,150,150,record_news)
        
        if result == 9:
            flap_save.save(save_path0 + name)
        elif result == 7:
            flap_save.save(save_path2 + name)
        elif result == 8:
            flap_save.save(save_path3 + name)
        elif result == 6:
            flap_save.save(save_path4 + name)
    else :
        print('pass')
        pass
#-----------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------写入excel----------------------------------------------------------
                  
#---------------------------------------------------------------------------------------------------------------------------------------



if __name__ == '__main__':
    path_txt='D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\record_news2.txt'
    record_news = open(path_txt,'w')
    path='D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\cut_masks\\'
    flap_path = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\binary_imgs\\'
    for name in listdir(path):
        if not name.startswith('.'):
            record_news.write( str(name) )
            print(name)
            img_in = Image.open( path + name )
            flap_in = Image.open( flap_path + name )
            img_array = np.array(img_in)
            flap_array = np.array(flap_in)
            Classify_one_or_two( img_array ,flap_array, name)
    
    # 单图调试        
    # path = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\cut_masks\\'
    # flap_path = 'D:\\资料\\work\\主动脉夹层\\U-net测试\\Pytorch-UNet-master\\first\\binary_imgs\\'
    # filename = '0791.png'
    # img_in = Image.open( path + filename)
    # flag_in = Image.open(flap_path +  filename)
    # img_array = np.array(img_in)
    # flap_array = np.array(flag_in)
    # Classify_one_or_two(img_array ,flap_array, filename, record_news)
    
    record_news.close()


