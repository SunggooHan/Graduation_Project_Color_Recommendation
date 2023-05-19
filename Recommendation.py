import numpy as np
import pandas as pd
import matplotlib.colors as cs
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import colorsys
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import matplotlib.colors as mcolors

# 이미지 로드
image = cv2.imread('example2.png')

# 초기 마스크 생성
mask = np.zeros(image.shape[:2], np.uint8)

# 배경과 전경 모델 정의
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 전경 객체가 포함된 관심 영역(ROI) 정의
rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

# GrabCut 세그멘테이션 수행
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 0과 2는 배경을 나타내고, 1과 3은 전경을 나타내는 마스크 생성
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 마스크를 원본 이미지에 적용
result = cv2.bitwise_and(image, image, mask=mask)

# 추출된 전경 이미지 저장
cv2.imwrite('foreground.png', result)

# 헬퍼 함수 정의
def hex2rgb(hex_value):
    h = hex_value.strip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return rgb

def rgb2hsv(r, g, b):
    return colorsys.rgb_to_hsv(r, g, b)

def kmeans_color_to_hsv(img, clusterno):
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = KMeans(n_clusters=clusterno)
    clt.fit(img)

    labels = np.unique(clt.labels_)
    hist, _ = np.histogram(clt.labels_, bins=np.arange(len(labels) + 1))

    colors = []
    hexlabels = []
    rgblabels = []
    hsvlabels = []

    for i in range(clt.cluster_centers_.shape[0]):
        normalized_color = clt.cluster_centers_[i] / 255.0
        colors.append(tuple(normalized_color))
        hexlabels.append(mcolors.to_hex(normalized_color))

    for i in range(len(hexlabels)):
        rgblabels.append(hex2rgb(hexlabels[i]))

    for i in range(len(rgblabels)):
        hsvlabels.append(rgb2hsv(rgblabels[i][0], rgblabels[i][1], rgblabels[i][2]))

    return hsvlabels

# 이미지 로드 및 처리
img = cv2.imread('foreground.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

clusterno = 5
final_hsvlabels = kmeans_color_to_hsv(img, clusterno)
original_hsv = final_hsvlabels[0]
print(original_hsv)

# CSV 파일 로드
df = pd.read_csv('ColorDataset.csv')

# H 값을 [0, 2pi]로 정규화
hsva_h = df["hsva/h"].to_numpy()
hsva_h_scaled = (hsva_h / 360) * (2 * np.pi)
hsva_h_scaled = hsva_h_scaled.reshape(-1, 1)

# S, V 값을 [0, 1]로 정규화
hsva_s = df["hsva/s"].to_numpy()
hsva_v = df["hsva/v"].to_numpy()

hsva_s_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(hsva_s.reshape(-1, 1))
hsva_v_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(hsva_v.reshape(-1, 1))

# 추출한 색상 정규화
hsv_h, hsv_s, hsv_v = original_hsv

hsv_h_scaled = (hsv_h / 360) * (2 * np.pi)
hsv_s_scaled = hsv_s / 100
hsv_v_scaled = hsv_v / 100

# 거리 계산에 사용할 데이터 구조 최적화
df_origin = pd.read_csv('ColorDataset.csv')
df_origin['hsva_h'] = hsva_h_scaled
df_origin['hsva_s'] = hsva_s_scaled
df_origin['hsva_v'] = hsva_v_scaled

# 거리 계산
distance = ((np.sin(hsv_h_scaled) * hsv_s_scaled * hsv_v_scaled - np.sin(df_origin['hsva_h']) * df_origin['hsva_s'] * df_origin['hsva_v']) ** 2
            + (np.cos(hsv_h_scaled) * hsv_s_scaled * hsv_v_scaled - np.cos(df_origin['hsva_h']) * df_origin['hsva_s'] * df_origin['hsva_v']) ** 2
            + (hsv_v_scaled - df_origin['hsva_v']) ** 2)

min_index = np.argmin(distance)
closest_color = df_origin.loc[min_index, 'key']
print(df_origin[df_origin['key'] == closest_color])