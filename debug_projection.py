import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load page 1 of source5
img = cv2.imread('data/images/source5/source5_p001.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# adaptive threshold
binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=25, C=10
)

# small kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
dilated = cv2.dilate(binary, kernel, iterations=1)

# projection
h_proj = dilated.sum(axis=1).astype(float)
h_proj_smooth = np.convolve(h_proj, np.ones(3)/3, mode='same')

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.imshow(dilated, cmap='gray')
ax1.set_title('Dilated binary')
ax2.plot(h_proj_smooth, range(len(h_proj_smooth)))
ax2.invert_yaxis()
ax2.axvline(h_proj_smooth.max() * 0.20, color='red', label='threshold')
ax2.set_title('Horizontal projection')
ax2.legend()
plt.tight_layout()
plt.savefig('data/crops/projection_debug.png', dpi=80)
print(f'Image height: {img.shape[0]}px')
print(f'Max projection: {h_proj_smooth.max():.0f}')
print(f'Threshold at 20%: {h_proj_smooth.max()*0.20:.0f}')
print('Saved: data/crops/projection_debug.png')
