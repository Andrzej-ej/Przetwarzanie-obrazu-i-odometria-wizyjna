import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_harris(img_gray, ksize):

    # pochodne kierunkowe operator Sobela
    Ix = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    # Rozmycie iloczynów filtrem Gaussa
    Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), 0)
    Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), 0)
    Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), 0)
    
    # Wyliczenie wyznacznika i śladu macierzy M 
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    
    # Obliczenie wartości H (WZÓR H(x,y) )

    k = 0.003 # Stała (im większa tym więcej punktów na obrazku)
    H = det_M - k * (trace_M ** 2)
   
    H_norm = cv2.normalize(H, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    return H_norm



# Wczytanie obrazu
img = cv2.imread('fontanna1.jpg')

img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# rozmiar maski = 3
H_map = calculate_harris(gray, ksize=3)
    
window_size = 8  # Większe okno oddala punkty od siebie
local_max = cv2.dilate(H_map, np.ones((window_size, window_size)))    
threshold = 0.02  # Wyższy próg filtruje słabsze narożniki
corner_mask = (H_map == local_max) & (H_map > threshold)

y_coords, x_coords = np.where(corner_mask)

# Rysowanie punktów
for y, x in zip(y_coords, x_coords):
    cv2.circle(img_disp, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)  # Czerwone kropki

plt.figure(figsize=(10, 8))
plt.imshow(img_disp)
plt.title('Wykryte narożniki metodą Harrisa')
plt.axis('off')
plt.show()