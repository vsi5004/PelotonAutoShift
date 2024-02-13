import cv2
import pytesseract
import cv2.aruco as aruco
import numpy as np

# Load the image
file_path = './img/photo2.jpg'
image = cv2.imread(file_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Scale the image to 50% of its original size
scaled_width = int(gray_image.shape[1] * 0.1)
scaled_height = int(gray_image.shape[0] * 0.1)
scaled_dimensions = (scaled_width, scaled_height)

# Use INTER_AREA interpolation for shrinking the image
scaled_image = cv2.resize(gray_image, scaled_dimensions, interpolation=cv2.INTER_AREA)

# Increase contrast by applying Histogram Equalization
contrast_enhanced_image = cv2.equalizeHist(scaled_image)

# Define the aruco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Detect aruco markers in the image
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

corners, ids, rejects = aruco_detector.detectMarkers(contrast_enhanced_image)

# Draw detected markers on the image
aruco.drawDetectedMarkers(contrast_enhanced_image, corners, ids)

# Calculate the angle between the centers of markers 0 and 1
if len(corners) >= 2:
    marker0_center = np.mean(corners[0][0], axis=0)
    marker1_center = np.mean(corners[1][0], axis=0)
    angle = np.arctan2(marker1_center[1] - marker0_center[1], marker1_center[0] - marker0_center[0]) * 180 / np.pi

    # Rotate the image by the calculated angle
    rows, cols = contrast_enhanced_image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(contrast_enhanced_image, M, (cols, rows))

    # Draw a line between the centers of the two markers with IDs 0 and 1
    cv2.line(rotated_image, tuple(map(int, marker0_center)), tuple(map(int, marker1_center)), (0, 255, 0), 2)

    # Calculate the width and height of the rectangle
    line_length = np.linalg.norm(marker1_center - marker0_center)
    rectangle_width = int(line_length * 0.6)
    rectangle_height = int(line_length * 0.6)

    # Calculate the center point of the rectangle
    rectangle_center = tuple(map(int, (marker0_center + marker1_center) / 2))

    # Calculate the top-left corner of the rectangle
    rectangle_top_left = (rectangle_center[0] - rectangle_width // 2, rectangle_center[1] - rectangle_height)

    # Calculate the bottom-right corner of the rectangle
    rectangle_bottom_right = (rectangle_center[0] + rectangle_width // 2, rectangle_center[1] - int(line_length * 0.2))

    # Draw the rectangle on the rotated image
    cv2.rectangle(rotated_image, rectangle_top_left, rectangle_bottom_right, (255, 0, 0), 2)
    
    # Crop the image to the rectangle
    cropped_image = rotated_image[rectangle_top_left[1]:rectangle_bottom_right[1], rectangle_top_left[0]:rectangle_bottom_right[0]]

    # Display the cropped image
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Invert the rotated image to make the text white and the background black
    inverted_image = 255 - cropped_image

    # Apply binary threshold to the inverted image
    _, thresholded_image = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Apply Gaussian blur to remove noise
    blurred_image = cv2.GaussianBlur(thresholded_image, (5, 5), 0)

    # Apply binary threshold to the blurred image
    _, final_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Display the processed image in a window
    cv2.imshow('Processed Image', final_image)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    cv2.destroyAllWindows()  # Destroy all windows after key press

    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(final_image, config='--psm 12')
    # Remove non-numeric characters from the text
    numeric_text = ''.join(filter(str.isdigit, text))

    # Print the numeric text
    print(numeric_text)

