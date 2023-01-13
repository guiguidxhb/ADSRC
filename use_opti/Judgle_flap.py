# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:45:58 2021

@author: admin
"""

import numpy as np
import math
import cv2
from PIL import Image
from predict import flip_predict
import os

slice_cnt = 0

#================================================判断是圆吗==========================================================
def Judge_circle(line): 
    s_area=cv2.contourArea(line[0])                                            #CV2库函数计算封闭曲线面积
    #list[0]是轮廓的列表集，list[0] [0]是列表集的第一个元素[[h,l]],故第三个[0]消列表括号，第四个[]则调用或者列
    #故调用方式为list[0][pin][0][h/l]    非常蠢，后面降维了
    pin_sum = len(line[0])
    line_array = np.asarray(line)
    line_array = line_array.ravel()                                            #数组降维,前面的调用方式太蠢了！！！
    line_array = line_array.reshape(-1,2)           #降维后重塑数组，此时调用形式为line_array[pin][lie=0/hang=1]   python牛皮！  注意这个轮廓[列x，行y]
    center_h,center_l = Find_center_circle(line_array,pin_sum)
    R_sq_max=0
    for p in range(pin_sum):
        R_sq =( line_array[p][1]-center_h )**2+( line_array[p][0]-center_l )**2
        if R_sq > R_sq_max:
          R_sq_max = R_sq
    s_max = math.pi * R_sq_max
    ratio_s = s_max/s_area
    
    if ratio_s >=2 :
        if s_area>5000:
            
            return 2 , line_array
        else:
            
            return 1 , line_array
    else :
        
        return 1 , line_array
    # cv2.drawContours(img_in,line,0,(0,0,255),3)  
    # print(line[0][0][0][0],type(line),len(line[0]))
#======================================================================================================================
#===================================================寻找轮廓中心========================================================
def Find_center_circle(line_array,pin_sum):
    h_pos = 0
    l_pos = 0
    for i in range(pin_sum):
        h_pos += line_array[i][1]
        l_pos += line_array[i][0]
    center_h = float(h_pos)/pin_sum
    center_l = float(l_pos)/pin_sum
    return center_h,center_l
#=======================================================================================================================
#=============================================求取轮廓最小矩形框并给出分割线==============================================
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
#======================================================================================================================================
#=======================================================数白色像素点的个数==============================================================
#需输入黑白图像的矩阵形式，k为白色连通域的数量，k=1时，仅有line_array[0]；k=2时，有line_array[0/1]。line_array为白色连通域的轮廓列表，其元素为边缘行列值[lie,hang]
def Judge_white( img_array , k , line_array , center_h0, center_h1, one_circle_threshold , threshold_0 , threshold_1 ,area0=[0],area1=[0]):     
     [h,l] = img_array.shape
     if k==1 :                                                  #单区域且为圆时           注意 line_array中的元素[lie，hang]
        # white_list=[]
        # find_flag = 0
        # for i in range(h):
        #     for j in range(l):
        #          if [j,i] in line_array and [j+1,i] not in line_array:   #确保从内边界开始搜索白块数量
        #             find_flag = 1
        #             a=1
        #             while [j+a,i] not in line_array  and j+a < l :
        #                 if img_array[i][j+a]==255 :
        #                     white_list.append([i,j+a])
        #                 else :
        #                     pass
        #                 a += 1
        #             break
        #     if find_flag == 1 and j == l-1 :
        #         break
        temp_array = img_array//255
        white_num = np.sum(temp_array)
        score_ascending_aorta = 0
        score_descending_aorta = sigmoid_threshold( float(white_num) )               
        
     elif k==2 :                                               #双区域                   注意line_array[0/1]中的元素为[lie,hang]
         # white_list_0 = []
         # white_list_1 = []
         # white_list = []
         # for i in range(h) :
         #     for j in range(l) :
         #         # print( [j,i] in line_array[0] and [j+1,i] not in line_array[0] )
         #         if ([j,i] in line_array[0]) and ([j+1,i] not in line_array[0]):   #确保从内边界开始搜索白块数量
         #             a=1
         #             # print('s',line_array[0])
         #             while ([j+a,i] not in line_array[0]) and (j+a < l) :
         #                 # print('0区域边缘点：',[j+a,i],img_array[i][j+a])
         #                 if img_array[i][j+a]==255:
         #                     white_list_0.append([i,j+a])
         #                 else :
         #                    pass
         #                 a += 1
         #             break
                 
         #         if ([j,i] in line_array[1]) and ([j+1,i] not in line_array[1]):   #确保从内边界开始搜索白块数量
         #             a=1
         #             while ([j+a,i] not in line_array[1]) and (j+a < l) :
         #                 if img_array[i][j+a]==255:
         #                     white_list_1.append([i,j+a])
         #                 else :
         #                     pass
         #                 a += 1
         #             break
         # area =[[minX,minY],[maxX,maxY]]
         box0 = img_array[area0[0][1]:area0[1][1]+1,area0[0][0]:area0[1][0]+1] // 255
         box1 = img_array[area1[0][1]:area1[1][1]+1,area1[0][0]:area1[1][0]+1] // 255
        
         num0 = np.sum(box0)
         num1 = np.sum(box1)    
     
         up_num = 0
         down_num = 0
         if center_h0 <= center_h1:
             up_num = num0
             threshold_up = threshold_0
             down_num = num1
             threshold_down = threshold_1
         else:
             up_num = num1
             threshold_up = threshold_1
             down_num = num0
             threshold_down = threshold_0
            
         score_ascending_aorta = sigmoid_threshold( float(up_num) )
         score_descending_aorta = sigmoid_threshold( float(down_num) )
     return score_ascending_aorta , score_descending_aorta
#================================================================================================================================
#=====================================================统计椭圆内两区域的白像素点个数===============================================
def Judge_white_oval(img_array,line_array,k,b,threshold,edge_array=[0]):
        # [h,l] = img_array.shape
        # white_list_up = []         #在分割线上
        # white_list_down =[]        #在分割线下
        # white_list_kb =[]          #正好在分割线上
        # find_flag = 0
        # for i in range(h):
        #     for j in range(l):
        #          # print(line_array)
        #          #print([j,i])
        #          if [j,i] in line_array and [j+1,i] not in line_array:   #确保从内边界开始搜索白块数量
        #             find_flag = 1
        #             a=1
        #             #print('区域数值',i-k*(j+a)-b)
        #             while [j+a,i] not in line_array  and j+a < l :
        #                 if img_array[i][j+a]==255 :
        #                     if i-k*(j+a)-b < 0 :
        #                        white_list_up.append([i,j+a])
        #                     elif i-k*(j+a)-b > 0 :
        #                         white_list_down.append([i,j+a])
        #                     else :
        #                         white_list_kb.append([i,j+a])
        #                 else :
        #                     pass
        #                 a += 1
        #             break
        #     if find_flag == 1 and j == l-1 :
        #         break
        # # print( len(white_list_up) , len(white_list_down) )
        # # record_news.write( '长条形白色块数量：  ' + str(len(white_list_up)) + '    '+ str(len(white_list_down)) +'\n')
        # if len(white_list_up) >= threshold and len(white_list_down) < threshold :             #  1  仅上区域多白点
        #     return 1
        # elif len(white_list_up) < threshold and len(white_list_down) >= threshold :           #  2  仅下区域多白点
        #     return 2
        # elif len(white_list_up) >= threshold and len(white_list_down) >= threshold :          #  3  上下两区域都有很多白点
        #     return 3
        # else :                                                                                #  4  上下两区域都没有白点
        #     return 4 
        up_num = 0
        down_num = 0
        # area =[[minX,minY],[maxX,maxY]]
        area = [np.min(edge_array[0],axis = 0),np.max(edge_array[0],axis = 0)]
        #print(area)
        for i in range(area[0][1],area[1][1] + 1):
            for j in range(area[0][0],area[1][0] + 1):
                if img_array[i][j] == 255:
                    if i-k*j-b <= 0 :
                        up_num += 1
                    else:
                        down_num += 1
        return np.array([up_num,down_num])
#===================================================================================================================================
#====================================================对单连通域和双连通域进行分类处理=================================================
def Classify_one_or_two(mask_in,flap_in):
    inf_list = cv2.findContours(mask_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    line = inf_list[1]                                                               #此处徐用的是inf_list[0],黄用的是inf_list[1]
    # print(line)
    num = len(line)
    # print(num)
    # print(num)
    # record_news.write( '连通区域'+str(num)+'\n' )
    # flap_save = Image.fromarray(flap_in.astype(np.uint8))
    # temp_img = np.ones((512,512))*0
    #cv2.drawContours(temp_img,line,1,(255,255,255),1)
    #cv2.drawContours(temp_img,line,0,(255,255,255),5)
    #cv2.imshow('temp_img',temp_img)
    #cv2.waitKey()
    
    # line_num = np.zeros(1)
    #print(line)
    
    del_index = np.array([-1])
    # print( '原line数：',len(line) )
    for i in range(0 , len(line)):
        if len(line[i]) <20:
            del_index = np.append(del_index,i)
        else:
            area = cv2.contourArea(line[i])
            if area < 300 :
                del_index = np.append(del_index,i)
    # print('del_index：',del_index)
    for i in range(1 , len(del_index)):
        j = len(del_index)-i
        del line[del_index[j]]
    num = len(line)
    
    score_ascending_aorta = 0
    score_descending_aorta = 0
    circle_flag = 0
    if num == 1 :
        circle_flag , line_array = Judge_circle(line)      #对于单连通区域，判断类圆形的形状区分为长条形和类圆形,这一步将line降维到line_array
        line_list = line_array.tolist()                           #将line转为array形式,为方便使用in函数判断点的列行值[j,i]是否在序列上
        
        if circle_flag==1 :                                       #是圆状时：找白块
            score_ascending_aorta , score_descending_aorta = Judge_white(flap_in,num,line_list, 0 , 0 ,150,150,150)
            
            
            
        elif circle_flag==2 :                                 #长条形参考价值不大，故暂不考虑
            pass
            # center_oval_h , center_oval_l = Find_center_circle( line_array,len(line_array) )
            # k , b = split(img_in,line[0],center_oval_l,center_oval_h)
            line_array = [ np.asarray(line[0]) ]
            line_array = [ line_array[0].ravel() ]
            line_array = [ line_array[0].reshape(-1,2) ]
            # result = Judge_white_oval( flap_in ,line_list, k , b ,150,line_array)
           
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
        area0 = [np.min(line_array[0],axis = 0),np.max(line_array[0],axis = 0)]
        area1 = [np.min(line_array[1],axis = 0),np.max(line_array[1],axis = 0)]
        score_ascending_aorta , score_descending_aorta = Judge_white(flap_in,num,line_list,center_h_0,center_h_1,150,150,150,area0,area1)
        
    else :
        # print('num=0')
        pass
    return score_ascending_aorta , score_descending_aorta , num , circle_flag
#-----------------------------------------------------------------------------------------------------------------------------------------------

def Calculate_score(mask_in,img_in,net,save_dir, slc_reverse, patient_name):
    '''统计连通域个数，去掉很小的连通域（干扰项）'''
    inf_list = cv2.findContours(mask_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    line = inf_list[1]                                                               #此处徐用的是inf_list[0],黄用的是inf_list[1]
    num = len(line)
    
    diameters = np.array((0, 0))
    del_index = np.array([-1])
    # print( '原line数：',len(line) )
    for i in range(0 , len(line)):
        if len(line[i]) <20:
            del_index = np.append(del_index,i)
        else:
            area = cv2.contourArea(line[i])
            if area < 300 :
                del_index = np.append(del_index,i)
    # print('del_index：',del_index)
    for i in range(1 , len(del_index)):
        j = len(del_index)-i
        del line[del_index[j]]
    num = len(line)
    
    '''计算得分'''
    score_ascending_aorta = 0
    score_descending_aorta = 0
    circle_flag = 0
    if num == 1 :
        circle_flag , line_array = Judge_circle(line)      #对于单连通区域，判断类圆形的形状区分为长条形和类圆形,这一步将line降维到line_array
        line_list = line_array.tolist()                           #将line转为array形式,为方便使用in函数判断点的列行值[j,i]是否在序列上
        #print(line_array)
        if circle_flag==1 :       
            area0 = [np.min(line_array,axis = 0),np.max(line_array,axis = 0)]                                #是圆状时：找白块
            score_ascending_aorta , score_descending_aorta, diameters = SegAndCnt(img_in,net,num,save_dir, slc_reverse, \
                                                                                  0 , 0 ,150,150,150,patient_name,area0, 0, line)
            
        elif circle_flag==2 :                                 
            pass

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
        area0 = [np.min(line_array[0],axis = 0),np.max(line_array[0],axis = 0)]
        area1 = [np.min(line_array[1],axis = 0),np.max(line_array[1],axis = 0)]
        score_ascending_aorta , score_descending_aorta, diameters = SegAndCnt(img_in,net, num,save_dir, slc_reverse,center_h_0,\
                                                                   center_h_1,150,150,150,patient_name,area0,area1, line)
        
    else :
        # print('num=0')
        pass
    return score_ascending_aorta , score_descending_aorta , num , circle_flag, diameters

def get_diameter(contour):
    ellipse = cv2.fitEllipse(contour)
    return ellipse

def SegAndCnt(img_array, net, area_num, save_dir, slc_reverse,center_h0, center_h1, one_circle_threshold , threshold_0 , threshold_1 ,\
              patient_name, area0=[0],area1=[0], contours = []):
    
    diameters = np.array((0,0))
    area_th = 0.93
    global slice_cnt
    if area_num == 1:
        box0 = img_array[area0[0][1]:area0[1][1]+1,area0[0][0]:area0[1][0]+1]
        AR0 = Image.fromarray(box0.astype(np.uint8))
        src_rows0 = area0[1][1] - area0[0][1]
        src_cols0 = area0[1][0] - area0[0][0]
        AR0 = AR0.resize((256,256))
        flip0 = Image.fromarray(flip_predict(AR0,net).astype(np.uint8))
        #flipImg = np.array(flip0.resize((src_rows0,src_cols0)))
        flipImg = np.array(flip0)
        #print(flipImg.shape)
        
        white_num = np.sum(flipImg)
        score_ascending_aorta = 0
        score_descending_aorta = sigmoid_threshold(float(white_num) )       
        
        #print(np.array(flip0).shape)
        #print(np.array(AR0).shape)
        
        img_show = np.concatenate([np.array(AR0), np.array(flip0) * 255], axis=1)
        img_save = Image.fromarray(img_show.astype(np.uint8))
        #img_save.save(save_dir +str(slc_reverse) + '.png')
        
        array_diameters = np.array(get_diameter(contours[0])[1])
        s_ellipse = math.pi*array_diameters[0]*array_diameters[1]/4
        x, y, w, h = cv2.boundingRect(contours[0])
        area = cv2.contourArea(contours[0])
        k = float(area)/(s_ellipse)
        if k >= area_th:
            diameters = array_diameters
            img_save_dir = os.path.join('diameters_down', patient_name)
            if not os.path.isdir(img_save_dir):
                os.makedirs(img_save_dir)
            score_ascending_aorta = sigmoid_threshold(float(white_num))
            if score_ascending_aorta == 1 and slice_cnt % 5 ==0:
                AR0.save(os.path.join(img_save_dir, str(slc_reverse) + '_down.png'))
                slice_cnt = 5
        slice_cnt += 1
        
        
    elif area_num == 2:
        box0 = img_array[area0[0][1]:area0[1][1]+1,area0[0][0]:area0[1][0]+1] 
        box1 = img_array[area1[0][1]:area1[1][1]+1,area1[0][0]:area1[1][0]+1] 
        
        AR0 = Image.fromarray(box0.astype(np.uint8))
        src_rows0 = area0[1][1] - area0[0][1]
        src_cols0 = area0[1][0] - area0[0][0]
        
        AR1 = Image.fromarray(box1.astype(np.uint8))
        src_rows1 = area1[1][1] - area1[0][1]
        src_cols1 = area1[1][0] - area1[0][0]
        
        AR0 = AR0.resize((256,256))
        AR1 = AR1.resize((256,256))
        
        flip0 = Image.fromarray(flip_predict(AR0,net).astype(np.uint8))
        flip1 = Image.fromarray(flip_predict(AR1,net).astype(np.uint8))
        # flipImg0 = np.array(flip0.resize((src_rows0,src_cols0)))
        # flipImg1 = np.array(flip1.resize((src_rows1,src_cols1)))
        
        flipImg0 = np.array(flip0)
        flipImg1 = np.array(flip1)
        
        num0 = np.sum(flipImg0)
        num1 = np.sum(flipImg1)    
     
        up_num = 0
        down_num = 0
        if center_h0 <= center_h1:
            up_num = num0
            threshold_up = threshold_0
            down_num = num1
            threshold_down = threshold_1
            
            array_diameters = np.array(get_diameter(contours[0])[1])
            s_ellipse = math.pi*array_diameters[0]*array_diameters[1]/4
            x, y, w, h = cv2.boundingRect(contours[0])
            area = cv2.contourArea(contours[0])
            k = float(area)/(s_ellipse)
            if k >= area_th:
                diameters = array_diameters
                img_save_dir = os.path.join('diameters', patient_name)
                if not os.path.isdir(img_save_dir):
                    os.makedirs(img_save_dir)
                score_ascending_aorta = sigmoid_threshold(float(up_num))
                if score_ascending_aorta == 1:
                    AR0.save(os.path.join(img_save_dir, str(slc_reverse) + '_up.png'))
            if slc_reverse == 453:
                print('k:',k, 'score_ascending_aorta')
        else:
            up_num = num1
            threshold_up = threshold_1
            down_num = num0
            threshold_down = threshold_0
            
            array_diameters = np.array(get_diameter(contours[1])[1])
            s_ellipse = math.pi*array_diameters[0]*array_diameters[1]/4
            x, y, w, h = cv2.boundingRect(contours[1])
            area = cv2.contourArea(contours[1])
            k = float(area)/(s_ellipse)
            if k >= area_th:
                diameters = array_diameters
                img_save_dir = os.path.join('diameters', patient_name)
                if not os.path.isdir(img_save_dir):
                    os.makedirs(img_save_dir)
                score_ascending_aorta = sigmoid_threshold(float(up_num))
                if score_ascending_aorta == 1:
                    AR1.save(os.path.join(img_save_dir, str(slc_reverse) + '_up.png'))
            if slc_reverse == 453:
                print('k:',k, 'score_ascending_aorta:', score_ascending_aorta)
                
        score_ascending_aorta = sigmoid_threshold( float(up_num) )
        if score_ascending_aorta == 0:
            diameters = np.array((0,0))
        score_descending_aorta = sigmoid_threshold( float(down_num) )
    
        img_show0 = np.concatenate([np.array(AR0), np.array(flip0) * 255], axis=1)
        img_show1 = np.concatenate([np.array(AR1), np.array(flip1) * 255], axis=1)
        
        img_save0 = Image.fromarray(img_show0.astype(np.uint8))
        # if center_h0 <= center_h1:
        #     img_save0.save(save_dir +str(slc_reverse) + '_up.png')
        # else:
        #     img_save0.save(save_dir +str(slc_reverse) + '_down.png')
        
        img_save1 = Image.fromarray(img_show1.astype(np.uint8))
        # if center_h0 <= center_h1:
        #     img_save1.save(save_dir +str(slc_reverse) + '_down.png')
        # else:
        #     img_save1.save(save_dir +str(slc_reverse) + '_up.png')
    else :
        pass

    return score_ascending_aorta , score_descending_aorta, diameters

def sigmoid_threshold(x):
    y = 1 if x > 162 else 0
    return y
    # y = 1.0/( 1 + math.exp(-x+100) )
    # return y

































