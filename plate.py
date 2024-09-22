import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

# Load the image
img = cv2.imread('C:/Users/hp/Downloads/ANPRwithPython-main/ANPRwithPython-main/image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Noise reduction
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
# Edge detection
edged = cv2.Canny(bfilter, 30, 200)

# Display the edges
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB))
plt.title('Edge Detection')
plt.axis('off')
plt.show()

# Find contours
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

if location is not None:
    # Create a mask for the detected license plate
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [location], -1, (255), -1)

    # Find bounding box for cropping
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    
    # Crop the image
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Display the cropped image
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB))
    plt.title('Cropped Image')
    plt.axis('off')
    plt.show()

    # OCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    if result:
        text = result[0][-2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw text on the original image
        res = cv2.putText(img, text=text, org=(location[0][0][0], location[0][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(res, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

        # Display the result image
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.title('Detected License Plate')
        plt.axis('off')
        plt.show()
    else:
        print("No text detected.")
else:
    print("No license plate detected.")
