import cv2 as c

# Load and resize image
img = c.imread(r'D:\Computer Vision\Image-Processing-Tutorials\Data\ironman.jpg')
img = c.resize(img, (900, 700))

# Convert to grayscale
gray = c.cvtColor(img, c.COLOR_BGR2GRAY)

# Invert the grayscale image
inverted = 255 - gray

# Apply Gaussian blur to the inverted image
blur = c.GaussianBlur(inverted, (21, 21), 0)

# Invert the blurred image
inverted_blur = 255 - blur

# Create the pencil sketch
pencil_sketch = c.divide(gray, inverted_blur, scale=256)

# Display original and sketch
c.imshow("Original", img)
c.imshow("Pencil Sketch", pencil_sketch)

# Wait until key is pressed
c.waitKey(0)
c.destroyAllWindows()

