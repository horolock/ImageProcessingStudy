import cv2
import matplotlib.pyplot as plt
from pathlib import Path

project_path = Path(__file__).parent.absolute()
image = cv2.imread(project_path/'images'/'like_lenna.png', cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def gaussian_pyramid(image, levels):
    pyramid = [image]

    for i in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    laplacian = []
    
    for i in range(len(gaussian_pyramid) - 1):
        next_level = cv2.pyrUp(gaussian_pyramid[i + 1])

        if next_level.shape[0] > gaussian_pyramid[i].shape[0]:
            next_level = next_level[:-1, :, :]
        if next_level.shape[1] > gaussian_pyramid[i].shape[1]:
            next_level = next_level[:, :-1, :]
        
        lap = cv2.subtract(gaussian_pyramid[i], next_level)

        laplacian.append(lap)

    laplacian.append(gaussian_pyramid[-1])
    return laplacian


# Example
levels = 5
g_pyramid = gaussian_pyramid(image_rgb, levels)
l_pyramid = laplacian_pyramid(g_pyramid)

min_height = min([img.shape[0] for img in l_pyramid])
concatenated_laplace_horizontal = cv2.resize(l_pyramid[0], (int(l_pyramid[0].shape[1] * min_height / l_pyramid[0].shape[0]), min_height))
fig, ax = plt.subplots(figsize=(15, 6))

for idx, img in enumerate(l_pyramid[1:], start=1):
    resized_img = cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height))
    concatenated_laplace_horizontal = cv2.hconcat([concatenated_laplace_horizontal, resized_img])

ax.imshow(concatenated_laplace_horizontal, cmap='gray')
ax.axis('off')

plt.title('Laplacian Pyramid')

plt.show()