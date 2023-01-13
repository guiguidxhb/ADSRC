# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:51:59 2023

@author: guiguidxhb
"""

import pydicom
import numpy as np
import SimpleITK as sitk
import dicom
import os
import pandas as pd


def get_pixel_spacing(dicom_dir):
    '''
        读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()  
    reader.LoadPrivateTagsOn()
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    
    img_num = 0
    need_pixel_spacing = 0
    for i in range (len(series_id)):
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id[i])
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
                    dcmFileInfo = pydicom.dcmread(dicom_names[0])
                    ## 获取像素数据
                    dcmPixelSpace = dcmFileInfo.PixelSpacing
                    need_pixel_spacing = dcmPixelSpace[0]
                    img_num = img_array.shape[0]
        
    return need_pixel_spacing


if __name__ == '__main__':
    diameters_dir = 'D:/资料/work/主动脉夹层/U-net测试/Pytorch-UNet-master/use_opti/diameters'
    cta_dir = 'D:/CTA/SUM/I'
    
    avg_diameters = []
    for patient_name in os.listdir(diameters_dir):
        if not os.path.isdir(os.path.join(diameters_dir, patient_name)) or patient_name == 'PENG GUANGMING_1':
            continue
        print(patient_name)
        xlsx_name = patient_name + '.xlsx'
        data = np.array(pd.read_excel(os.path.join(diameters_dir, xlsx_name)))[:,0:2]
        pixel_spacing = get_pixel_spacing(os.path.join(cta_dir, patient_name))
        #print(pixel_spacing)
        mimeters = data * pixel_spacing / (512/287)
        avg_d = mimeters.mean()
        
        cmb_data = np.hstack((data, mimeters))
        
        df = pd.DataFrame(cmb_data)
        df.to_excel(diameters_dir+ '/'+ patient_name + '.xlsx', index=False)
        
        avg_diameters.append([patient_name, avg_d])
        
    df = pd.DataFrame(avg_diameters)
    df.to_excel(diameters_dir+ '/' + 'avg_diameters.xlsx', index=False)
    
        
    