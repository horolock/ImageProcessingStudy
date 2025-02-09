from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

project_path = Path(__file__).parent.absolute()

image = cv2.imread(project_path/'images'/'like_lenna.png', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrrum_original = 20 * np.log(np.abs(fshift))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(magnitude_spectrrum_original, cmap='gray')
plt.title('Magnitude Spectrum')
plt.xticks([])
plt.yticks([])

plt.show()


# Low-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
radius = 30
mask = np.ones((rows, cols), np.uint8)
mask[crow - radius : crow + radius, ccol - radius : ccol + radius] = 0
fshift_filtered = fshift * mask

# Inverted fourier transform
f_ishift = np.fft.ifftshift(fshift_filtered)
image_back = np.fft.ifft2(f_ishift)
image_back = np.abs(image_back)

# Result image
plt.figure(figsize=(12, 12))

plt.subplot(133)
plt.imshow(image_back, cmap='gray')
plt.title('Reconstructed Image')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()