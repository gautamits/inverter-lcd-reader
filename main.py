import cv2
import numpy as np
import pytesseract
from capture_all import capture_all

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

def read_all():
    sprite = capture_all(2, 3)
    gray, thresh, blurred, segmented_image = get_gray_thresh(sprite)
    return cv2.vconcat([sprite, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)])

if __name__ == "__main__":
    sprite=read_all()
    # 5. Save the processed output image
    output_path = "detected_rectangles.jpg"
    cv2.imwrite(output_path, sprite)
