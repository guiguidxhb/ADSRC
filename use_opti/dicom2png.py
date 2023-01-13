import os
import SimpleITK as sitk
import dicom
import numpy as np
import cv2
from tqdm import tqdm

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

def normalize(image):
    Src_img = image
    
    MIN_BOUND = -150.0
    MAX_BOUND = 350.0
    
    image = (Src_img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    
    return image

if __name__ == '__main__':
    root_dir = 'D:/NNDS/2021.8.9/Normal_CTA/'
    dicom_dir = root_dir + 'AI BING XIANG/'
    # 读取dicom文件的元数据(dicom tags)
    #slices = load_patient(dicom_dir)
    #print('The number of dicom files : ', len(slices))
    # 提取dicom文件中的像素值
    image = get_pixels_hu_by_simpleitk(dicom_dir)
    num = 0
    for i in tqdm(range(image.shape[0])):
        #i = j//2
        num +=1 ;
        img_path = dicom_dir + "img/" + str(i).rjust(4, '0') + "_II7.png"
        #img_path_edge = root_dir +"正常png/" + "img2/" + str(i).rjust(4, '0') + "_II7.png"
        # 将像素值归一化到[0,1]区间
        for_edge , org_img = normalize_hu(image[i])
        # 保存图像数组为灰度图(.png)
        
        cv2.imwrite(img_path, org_img * 255)
        
        # if(i>=300 and i<=700 and i%5==0):
        #     cv2.imwrite(img_path, org_img * 255)
        #     cv2.imwrite(img_path_edge, for_edge * 255)


