from matplotlib import pyplot as plt
from matplotlib import image as mpimg
 
plt.title("Sheep Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")
 
image1 = mpimg.imread('./Track/Track-front/502.png')
image2 = mpimg.imread('./Track/Track-store/502.png')
fig, (ax1, ax2) = plt.subplots(1, 2)

# Display image1 in the first subplot
ax1.imshow(image1)
ax1.set_title('Image 1')

# Display image2 in the second subplot
ax2.imshow(image2)
ax2.set_title('Image 2')

# Show the plot
plt.show()
