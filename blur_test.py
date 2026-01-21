import cv2
import numpy as np

image = cv2.imread("reference.jpg")

if image is None:
    raise FileNotFoundError("reference.jpg not found in the main directory")

blur_light = cv2.GaussianBlur(image, (5, 5), 0)
blur_medium = cv2.GaussianBlur(image, (15, 15), 0)
blur_heavy = cv2.GaussianBlur(image, (31, 31), 0)

cv2.imwrite("blur_light.jpg", blur_light)
cv2.imwrite("blur_medium.jpg", blur_medium)
cv2.imwrite("blur_heavy.jpg", blur_heavy)

def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def tenengrad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return (gx ** 2 + gy ** 2).mean()

def fft_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude.mean()

print("\nREFERENCE IMAGE")
print("Laplacian Variance:", laplacian_variance(image))
print("Tenengrad:", tenengrad(image))
print("FFT Metric:", fft_blur(image))

print("\nLIGHT BLUR")
print("Laplacian Variance:", laplacian_variance(blur_light))
print("Tenengrad:", tenengrad(blur_light))
print("FFT Metric:", fft_blur(blur_light))

print("\nMEDIUM BLUR")
print("Laplacian Variance:", laplacian_variance(blur_medium))
print("Tenengrad:", tenengrad(blur_medium))
print("FFT Metric:", fft_blur(blur_medium))

print("\nHEAVY BLUR")
print("Laplacian Variance:", laplacian_variance(blur_heavy))
print("Tenengrad:", tenengrad(blur_heavy))
print("FFT Metric:", fft_blur(blur_heavy))
