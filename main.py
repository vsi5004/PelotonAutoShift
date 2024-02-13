import cv2

# Load the image
file_path = './img/photo1.jpg'
image = cv2.imread(file_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Scale the image to 50% of its original size
scaled_width = int(gray_image.shape[1] * 0.20)
scaled_height = int(gray_image.shape[0] * 0.20)
scaled_dimensions = (scaled_width, scaled_height)

# Use INTER_AREA interpolation for shrinking the image
scaled_image = cv2.resize(gray_image, scaled_dimensions, interpolation=cv2.INTER_AREA)

# Increase contrast by applying Histogram Equalization
contrast_enhanced_image = cv2.equalizeHist(scaled_image)

# Invert the image by subtracting from maximum pixel value (255)
inverted_image = 255 - contrast_enhanced_image

threshold_value, thresholded_image = cv2.threshold(inverted_image, 80, 255, cv2.THRESH_BINARY)

# Apply Gaussian blur to reduce noise
# The (5, 5) indicates the size of the Gaussian kernel; you can adjust these values
blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)

ret, smoothed_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY)


# Display the processed image in a window
cv2.imshow('Processed Image', smoothed_image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()  # Destroy all windows after key press


import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(thresholded_image, config='--psm 7')
print(text)