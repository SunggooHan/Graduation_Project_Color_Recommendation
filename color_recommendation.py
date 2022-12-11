import numpy as np
import pandas as pd
import matplotlib.colors as cs
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import sys
from  PIL  import Image
import colorsys

img_origin = cv2.imread('example.png', cv2.IMREAD_UNCHANGED)
original = img_origin.copy()

l = int(max(5, 6))
u = int(min(6, 6))

ed = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
edges = cv2.GaussianBlur(img_origin, (21, 51), 3)
edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(edges, l, u)

_, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

data = mask.tolist()
sys.setrecursionlimit(10**8)
for i in  range(len(data)):
    for j in  range(len(data[i])):
        if data[i][j] !=  255:
            data[i][j] =  -1
        else:
            break
    for j in  range(len(data[i])-1, -1, -1):
        if data[i][j] !=  255:
            data[i][j] =  -1
        else:
            break
image = np.array(data)
image[image !=  -1] =  255
image[image ==  -1] =  0

mask = np.array(image, np.uint8)

result = cv2.bitwise_and(original, original, mask=mask)
result[mask ==  0] =  255
cv2.imwrite('bg.png', result)

img = Image.open('bg.png')
img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img.putdata(newData)
img.save("img.png", "PNG")

# converting from BGR to RGB
img = cv2.imread("img.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# reshape the image
img=img.reshape((img.shape[1]*img.shape[0],3))

# KMeans
kmeans=KMeans(n_clusters=5)
s=kmeans.fit(img)

# labels
labels=kmeans.labels_
labels=list(labels)

# determine centroids of clusters
centroid=kmeans.cluster_centers_

percent=[]
for i in range(len(centroid)):
  j=labels.count(i)
  j=j/(len(labels))
  percent.append(j)

# plot a pie chart
plt.pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))

# https://intrepidgeeks.com/tutorial/extract-color-from-image

#set path to image
imgpath = 'img.png'
#set number of cluster for kmeans
clusterno = 5

#read image
img = cv2.imread(imgpath)
#convert bgr to rgb
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#reshape img array
n_img = np.reshape(img,(img.shape[0]*img.shape[1],3))

#use kmeans to find cluster of color
clt = KMeans(n_clusters=clusterno)
clt.fit(n_img)

#get unique value of labels in kmeans
labels = np.unique(clt.labels_)

#find the pixel numbers of each color that is set by cluster number
hist,_ = np.histogram(clt.labels_,bins=np.arange(len(labels)+1))

#declare list to hold color to be used in chart
colors = []

#declare list to hold hex color code for labeling in chart
hexlabels = []

#get the main color
for i in range(clt.cluster_centers_.shape[0]):
  colors.append(tuple(clt.cluster_centers_[i]/255))
  hexlabels.append(cs.to_hex(tuple(clt.cluster_centers_[i]/255)))

#create pie chart for color
plt.pie(hist,labels=hexlabels,colors=colors,autopct='%1.1f%%')
plt.axis('equal')

def hex2rgb(hex_value):
    h = hex_value.strip("#") 
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb

hex2rgb(hexlabels[0])

rgblabels = []
for i in range(len(hexlabels)):
    rgblabels.append(hex2rgb(hexlabels[i]))

def rgb2hsv(r,g,b):
    return colorsys.rgb_to_hsv(r,g,b)

hsvlabels = []
for i in range(len(rgblabels)):
    hsvlabels.append(rgb2hsv(rgblabels[i][0], rgblabels[i][1], rgblabels[i][2]))

# hsv (hue 색상, saturation 채도, value 명도)(v 대신 i,b,l 도 됨)
# H : 0 ~ 360
# S : 0 ~ 100 (진함)
# V : 0 ~ 100 (밝음)

#create pie chart for color
plt.pie(hist,labels=rgblabels,colors=colors,autopct='%1.1f%%')
plt.axis('equal')

def hex2rgb(hex_value):
    h = hex_value.strip("#") 
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def rgb2hsv(r,g,b):
    return colorsys.rgb_to_hsv(r,g,b)

def kmeans_color_to_hsv(img):

    # reshape the image
    img=img.reshape((img.shape[1]*img.shape[0],3))

    #use kmeans to find cluster of color
    clt = KMeans(n_clusters=clusterno)
    clt.fit(n_img)

    #get unique value of labels in kmeans
    labels = np.unique(clt.labels_)

    #find the pixel numbers of each color that is set by cluster number
    hist,_ = np.histogram(clt.labels_,bins=np.arange(len(labels)+1))

    #declare list to hold color to be used in chart
    colors = []

    #declare list to hold hex color code for labeling in chart
    hexlabels = []

    #declare list to hold rgb color code for labeling in chart
    rgblabels = []

    #declare list to hold hsv color code for labeling in chart
    hsvlabels = []

    #get the main color
    for i in range(clt.cluster_centers_.shape[0]):
        colors.append(tuple(clt.cluster_centers_[i]/255))
        hexlabels.append(cs.to_hex(tuple(clt.cluster_centers_[i]/255)))
        
    for i in range(len(hexlabels)):
        rgblabels.append(hex2rgb(hexlabels[i]))

    for i in range(len(rgblabels)):
        hsvlabels.append(rgb2hsv(rgblabels[i][0], rgblabels[i][1], rgblabels[i][2]))

    return hsvlabels

#read image
img = cv2.imread('img.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

final_hsvlabels = kmeans_color_to_hsv(img)

original_hsv = final_hsvlabels[0]
print(original_hsv)

#정규화
df = pd.read_csv('ColorDataset.csv')
hsva_h = df["hsva/h"]
hsva_s = df["hsva/s"]
hsva_v = df["hsva/v"]

#h값 [0,2pi] 정규화
df = pd.read_csv('ColorDataset.csv')
hsva_h = df["hsva/h"].to_numpy()
hsva_h = hsva_h.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
from math import pi

scaler = MinMaxScaler(feature_range=(0,2*pi))
scaler.fit(hsva_h)
hsva_h_scaled = scaler.transform(hsva_h)

hsva_h = pd.DataFrame(hsva_h_scaled, columns=['hsva_h'])

#sv값 [0,1] 정규화
from sklearn.preprocessing import minmax_scale

hsva_s = pd.DataFrame(data=minmax_scale(hsva_s))
hsva_v = pd.DataFrame(data=minmax_scale(hsva_v))

hsva_s.columns = ["hsva_s"]
hsva_v.columns = ["hsva_v"]

scaled_hsv_set = hsva_h.join(hsva_s).join(hsva_v)

#(sin(h1)*s1*v1 - sin(h2)*s2*v2 )^2 + (cos(h1)*s1*v1 - cos(h2)*s2*v2)^2 + ( v1 - v2 )^2

hsv = original_hsv
hsv_h = hsv[0]
hsv_s = hsv[1]
hsv_v = hsv[2]

#추출한 h 값 정규화
hsv_h = (hsv_h/360)*2*pi

#추출한 s값 정규화
hsv_s = hsv_s/100

#추출한 s값 정규화
hsv_v = hsv_v/100

scaled_hsv = [hsv_h, hsv_s, hsv_v]

#거리계산
import math

hsv_h = scaled_hsv[0]
hsv_s = scaled_hsv[1]
hsv_v = scaled_hsv[2]

df = scaled_hsv_set
df_origin = pd.read_csv('ColorDataset.csv')
hsva_h = df["hsva_h"].to_numpy()
hsva_s = df["hsva_s"].to_numpy()
hsva_v = df["hsva_v"].to_numpy()
dist = []

for i in range(len(hsva_h)):
    distance = ((math.sin(hsv_h)*hsv_s*hsv_v)-(math.sin(hsva_h[i])*hsva_s[i]*hsva_v[i]))*((math.sin(hsv_h)*hsv_s*hsv_v)-(math.sin(hsva_h[i])*hsva_s[i]*hsva_v[i]))
    + ((math.cos(hsv_h)*hsv_s*hsv_v)-(math.cos(hsva_h[i])*hsva_s[i]*hsva_v[i]))*((math.cos(hsv_h)*hsv_s*hsv_v)-(math.cos(hsva_h[i])*hsva_s[i]*hsva_v[i]))
    + (hsv_v-hsva_v[i])*(hsv_v-hsva_v[i])
    
    dist.append(distance)

min(dist)
num = dist.index(min(dist))

print(df_origin.loc[df_origin['key'] == df_origin.loc[num, 'key']])