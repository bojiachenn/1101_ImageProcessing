import cv2
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.exposure import match_histograms

# 圖片讀取
image = cv2.imread('img/GT86.jpg')
multi = True if image.shape[2] > 1 else False

reference = cv2.imread('img/S2000.jpg')

# 將原圖轉為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

# histogram equalization
he = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", he)

# Histogram Matching
matched = match_histograms(image, reference, multichannel=multi)
matched_gray = cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY)
cv2.imshow("matched", matched_gray)

mhe = cv2.equalizeHist(matched_gray)

# clahe
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
clahe_img = clahe.apply(gray)
cv2.imshow("clahe", clahe_img)

# 儲存
cv2.imwrite("1_gray.jpg", gray)
cv2.imwrite("2_he.jpg", he)
cv2.imwrite("3_clahe.jpg", clahe_img)
cv2.imwrite("4_matched.jpg", matched)
cv2.imwrite("5_matched_gray.jpg", matched_gray)
cv2.imwrite("6_reference_gray.jpg", reference_gray)
cv2.imwrite("7_mhe.jpg", referencmhee_gray)

# Original HE 的比較
fig, he_axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
for i, img in enumerate((gray, he)):
    img_hist, bins = exposure.histogram(img[...], source_range='dtype')
    he_axes[i].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(img[...])
    he_axes[i].plot(bins, img_cdf)
    he_axes[0].set_ylabel('gray')

he_axes[0].set_title('Original')
he_axes[1].set_title('Histogram Equalization')

plt.savefig('Histogram Equalization.png')
plt.show()

# Original Clahe 的比較
fig, clahe_axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
for i, img in enumerate((gray, clahe_img)):
    img_hist, bins = exposure.histogram(img[...], source_range='dtype')
    clahe_axes[i].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(img[...])
    clahe_axes[i].plot(bins, img_cdf)
    clahe_axes[0].set_ylabel('gray')

clahe_axes[0].set_title('Original')
clahe_axes[1].set_title('Clahe')

plt.savefig('clahe.png')
plt.show()

# Original HM 的比較
fig, hm_axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, img in enumerate((gray, reference_gray, matched_gray)):
    
    img_hist, bins = exposure.histogram(img[...], source_range='dtype')
    hm_axes[i].plot(bins, img_hist / img_hist.max())
    img_cdf, bins = exposure.cumulative_distribution(img[...])
    hm_axes[i].plot(bins, img_cdf)
    hm_axes[0].set_ylabel('gray')

hm_axes[0].set_title('Original')
hm_axes[1].set_title('Reference')
hm_axes[2].set_title('Histogram Matched')

plt.tight_layout()
plt.savefig('comparison.png')
plt.show()

if (cv2.waitKey(0)==27):
    cv2.destroyAllwindows()
