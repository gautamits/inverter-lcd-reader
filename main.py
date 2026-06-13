import time
import cv2
import numpy as np
from picamera2 import Picamera2
import pytesseract
WIDTH=1280
HEIGHT=720
CONTOUR_OFFSET=10

picam = Picamera2()
picam.configure(picam.create_still_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)}))
picam.start()
    
print("Camera warming up...")
time.sleep(2) 

def get_contour(edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0

    # 4. Filter contours to find right-angle rectangles
    right_contours=[]
    for contour in contours:
        # Ignore very small artifacts/noise
        if cv2.contourArea(contour) < 500:
            continue

        # Calculate perimeter and approximate the contour shape
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # A rectangle must have exactly 4 vertices
        if len(approx) == 4:
            # Verify right angles by checking the aspect ratio stability or 
            # leveraging bounding box alignment versus shape area.
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            contour_area = cv2.contourArea(approx)

            # If the shape closely matches its bounding box, it's a right-angled rectangle
            if abs(1.0 - (contour_area / rect_area)) < 0.15:
                # Draw the contour on the original BGR image in Red (B=0, G=0, R=255)
                # Thickness is set to 3 pixels
                # cv2.drawContours(img_bgr, [approx], -1, (0, 0, 255), 3)
                right_contours.append(contour)
                rect_count += 1
    print(f"Found {rect_count} right-angled rectangles.")
    if right_contours:
        largest_contour = max(right_contours, key=cv2.contourArea)
        return largest_contour
    return None

def get_edged():
    # Capture image directly into a NumPy array
    print("Capturing image...")
    image = picam.capture_array()

    # Convert RGB array from PiCamera2 to OpenCV's native BGR format
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 2. Pre-process the image for contour detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged, gray, blurred, img_bgr

def k_means_clustering(gray, k=100):
    # Reshape the image to a 2D array of pixels and 1 color value (grayscale)
    pixel_values = gray.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and reshape to original image dimensions
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(gray.shape)
    return segmented_image

def k_means_clustering_color(img, k=5):
    pixels = np.float32(img.reshape(-1, 3))

    # Define criteria and number of clusters (groups)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5  # Number of dominant groups you want

    # Cluster the pixels
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count how many pixels fell into each group
    _ , counts = np.unique(labels, return_counts=True)

    # Map counts to their respective cluster center colors
    dominant_groups = zip(centers, counts)
    return dominant_groups
def capture_cropped(contour):
    edged, gray, blurred, img_bgr = get_edged()
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = img_bgr[y+CONTOUR_OFFSET:y+h-CONTOUR_OFFSET, x+CONTOUR_OFFSET:x+w-CONTOUR_OFFSET]
    return cropped_image

def get_text(img):
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789vA.'
    text = pytesseract.image_to_string(img, config=custom_config)
    print(f"Extracted Text: {text}\n")

def get_gray_thresh(cropped):
    gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    segmented_image = k_means_clustering(gray_image)
    blurred = cv2.bilateralFilter(
        gray_image,
        5, # Diameter of each pixel neighborhood used during filtering. A larger value means that more distant pixels will influence each other, resulting in stronger smoothing.
        40, # SigmaColor: Filter sigma in the color space. A larger value means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color.
        40 # SigmaSpace: Filter sigma in the coordinate space. A larger value means that farther pixels will influence each other, resulting in stronger smoothing.
    )

    thresh_img = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Use a Gaussian-weighted sum of the neighborhood values minus the constant to determine the threshold for each pixel. This can help handle varying lighting conditions across the image.
        cv2.THRESH_BINARY_INV, # Invert the binary image so that text becomes white and background becomes black, which can improve OCR accuracy for LCD segments.
        13, # Block size (must be odd). Determines the size of the neighborhood used to calculate the threshold for each pixel. A common choice is 11 or 15.
        2 # Constant subtracted from the mean or weighted mean. Adjust this value to fine-tune the thresholding effect.
    )
    return gray_image, thresh_img, blurred, segmented_image

def manage_histogram(gray):
    n_bins = 16

    # 3. Calculate the scale factor
    bin_size = 256 / n_bins

    # 4. Convert to binned pixels
    # floor() groups the values, astype(np.uint8) ensures a valid image format
    binned_image = np.floor(gray / bin_size) * (255 / (n_bins - 1))
    binned_image = binned_image.astype(np.uint8)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])

    # 3. Find the most frequent pixel intensity (the peak)
    most_frequent_pixel = np.argmax(hist)

    # 4. Define your range around the peak (e.g., +/- 10)
    # Ensure values do not go below 0 or above 255
    lower_bound = max(0, most_frequent_pixel - 10)
    upper_bound = min(255, most_frequent_pixel + 10)

    # 5. Create a mask for pixels within this most frequent range
    mask = (hist >= lower_bound) & (hist <= upper_bound)

    # 6. Subtract (zero out) the frequent range
    # This makes those pixels black (0) and leaves the rest of the image untouched
    result_img = hist.copy()
    result_img[mask] = 0
    return result_img

def capture_all(img_count, time_gap):
    edged, gray, blurred, img_bgr = get_edged()
    largest_contour = get_contour(edged)
    if largest_contour is None:
        print("No right-angled rectangles found.")
        cv2.imwrite('error.jpg', img_bgr)
        exit()
    cropped=capture_cropped(largest_contour)
    gray, thresh, blurred, segmented_image = get_gray_thresh(cropped)
    # sub = manage_histogram(gray)
    [gray_3ch, thresh_3ch, blurred_3ch, segmented_3ch] = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), [gray, thresh, blurred, segmented_image]))
    print(get_text(thresh))
    result=cv2.vconcat([cropped, gray_3ch, blurred_3ch, segmented_3ch, thresh_3ch])
    time.sleep(time_gap)
    for index, value in enumerate(range(img_count), start=0):
        try:
            cropped = capture_cropped(largest_contour)
            gray, thresh, blurred, segmented_image = get_gray_thresh(cropped)
            # sub = manage_histogram(gray)
            print(get_text(thresh))
            [gray_3ch, thresh_3ch, blurred_3ch, segmented_3ch] = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), [gray, thresh, blurred, segmented_image]))
            result=cv2.hconcat([result, cv2.vconcat([cropped, gray_3ch, blurred_3ch, segmented_3ch, thresh_3ch])])
        except Exception as E:
            print(f"Error capturing image {index}: {E}")
            pass
        time.sleep(time_gap)
    return result

if __name__ == "__main__":
    sprite=capture_all(2, 3)
    picam.stop()
    # 5. Save the processed output image
    output_path = "detected_rectangles.jpg"
    cv2.imwrite(output_path, sprite)
