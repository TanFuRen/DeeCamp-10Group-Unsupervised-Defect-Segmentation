from skimage.measure import compare_ssim
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

original_path = 'C:/Users/lmy/Desktop/DeeCamp/segmentation/data/broken_large/ori483.jpg' #原始图像路径
reconstructive_path = 'C:/Users/lmy/Desktop/DeeCamp/segmentation/data/good/gen483.jpg'  #重建图像路径

class Segmentation(object):
    def __init__(self, ori_path, re_path):
        self.ori_path = ori_path
        self.re_path = re_path
    
    def load_image(self):
        """
        load the two input images

        return
        a: original image
        a_hat: reconstructive image
        """
        a = cv2.imread(self.ori_path) 
        a_hat = cv2.imread(self.re_path)
        return a,a_hat
    
    def convert2gray(self, image_a, image_b):
        """
        convert the images to grayscale

        input:
        image_a: original image
        image_b: reconstructive image
        return:
        a_gray: convert original image to gray
        a_hat_gray: convert reconstructive image to gray
        """
        a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        a_hat_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
        return a_gray,a_hat_gray
    
    def compute_ssim(self, gray_a, gray_b):
        """
        compute the Structural Similarity Index (SSIM) between the two images

        input:
        gray_a: gray image of the original image
        gray_b: constucrtive image of the original image
        return:
        s: The value of ssim
        d: the difference image
        """
        (s, d) = compare_ssim(gray_a, gray_b, full=True)
        d = (d * 255).astype("uint8")
        return s, d

    def Binary_processing(self, threhold, d1):
        """
        Set threshold value, binary value processing

        input:
        threhold: 阈值
        d1: the difference image
        return: 
        m: mask   
        """
        m = d1.copy()
        m[m < threhold] = 0   #小于阈值的为0，即可能不是缺陷区域
        m[m >= threhold] = 255  #大于阈值的为255，即可能是缺陷区域
        return m
    
    def plot(self, ori, re, m1):
        """
        input:
        ori: original image
        re: reconstructive image
        m1: mask
        """
        ori = cv2.resize(ori,(200,200),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('original',ori)  #原始图像

        re =  cv2.resize(re,(200,200),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('reconstruction',re)  #重建图像

        m1 = cv2.resize(m1,(200,200),interpolation=cv2.INTER_CUBIC)  
        cv2.imshow('mask',m1)  #mask
        # plt.subplot(111)
        # plt.imshow(m1, plt.cm.gray)
        
        m1 = np.array(m1 ,np.uint8)
        contours, hierarchy = cv2.findContours(m1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        middle = ori.copy()
        result = cv2.drawContours(middle,contours,-1,(0,0,255),3)
        result = cv2.resize(result,(200,200),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('result',result) #轮廓

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    seg = Segmentation(original_path, reconstructive_path)
    x, x_hat = seg.load_image()
    x_gray, x_hat_gray = seg.convert2gray(x, x_hat)
    score, diff = seg.compute_ssim(x_gray, x_hat_gray)
    mask = seg.Binary_processing(36, diff)
    seg.plot(x, x_hat, mask)
