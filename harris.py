import cv2 as cv
import numpy as np
import os
import math
from scipy import ndimage

img = cv.imread('sample4.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)


def getParams(grad_x,grad_y,x,y):
    a = b = c = 0.00
    
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            a = a + (grad_x[i][j])**2
            b = b + grad_y[i][j]**2
            c = c + grad_x[i][j]*grad_y[i][j]
    return a,2*b,c

def NMS(corner_image,dp,n,m):
    tr = [[0]*m for _ in range(n)]
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            if dp[i][j]>=dp[i-1][j] and dp[i][j]>=dp[i-1][j-1] and dp[i][j]>=dp[i-1][j+1] and dp[i][j]>=dp[i][j-1] and dp[i][j]>=dp[i][j+1] and dp[i][j]>=dp[i+1][j] and dp[i][j]>=dp[i+1][j-1] and dp[i][j]>=dp[i+1][j+1]:
                if dp[i][j]>0:
                    cv.circle( corner_image, (i, j), 3, (255,0,0), 3)
    
    return corner_image
            
def HarrisCorner(img,threshold,k):
    n,m,h = img.shape
    print(n,m)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = ndimage.gaussian_filter(gray, sigma = 1.0)
    grad_x = cv.Sobel(gray,cv.CV_16S,1,0,0,borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray,cv.CV_16S,0,1,0,borderType=cv.BORDER_DEFAULT)
    grad_x = grad_x.astype('float64')
    grad_y = grad_y.astype('float64')
    
    dp = [[-1]*m for _ in range(n)]
    
    mn = 0
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            a,b,c = getParams(grad_x,grad_y,i,j)
            lambda1 = a+c+ (math.sqrt(b**2 + (a-c)**2))
            lambda1 = lambda1/2
            
            lambda2 = a+c - (math.sqrt(b**2 + (a-c)**2))
            lambda2 = lambda2/2

            R = lambda1*lambda2 - ((lambda1-lambda2)**2)*k
            mn = min(mn,R)
            dp[i][j] = R

    for i in range(n):
        for j in range(m):
            if dp[i][j] == -1:
                dp[i][j] = mn
    return dp       

n,m,h = img.shape
dp = HarrisCorner(img,5,0.04)
corner_image = np.copy(img)
corner_image = NMS(corner_image,dp,n,m)
cv.imshow('img',corner_image)
cv.waitKey(0)
cv.imwrite('Corners_Detected.png',corner_image)




