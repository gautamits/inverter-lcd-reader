# pip install opencv-python pytesseract numpy
# Also install tesseract binary on your system:
#   macOS: brew install tesseract
#   Ubuntu/RPi: sudo apt-get install tesseract-ocr

import cv2
import numpy as np
import pytesseract
import re
import os
import pandas as pd

def detect_edges(blurred):
    # v = np.median(blurred)
    # lower = int(max(0, (1.0 - 0.33) * v))
    # upper = int(min(255, (1.0 + 0.33) * v))
    # print(f"Edge detection thresholds: lower={lower}, upper={upper}")
    # canny_edges = cv2.Canny(blurred, lower, upper)
    canny_edges = cv2.Canny(blurred, 50, 100)

    # 3. Sobel Edge Detection (Calculates directional gradients)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3) # Horizontal gradients
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3) # Vertical gradients
    sobel_combined = cv2.bitwise_or(cv2.convertScaleAbs(sobel_x), cv2.convertScaleAbs(sobel_y))

    return canny_edges, sobel_combined

def detect_digits(gray):
    gray = np.ascontiguousarray(gray)

    # 2. Thresholding to binarize the image (Otsu's method works best)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = get_closed_image(thresh)  # Morphological closing to fill gaps
    # 3. Find contours of the individual numbers
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), "contours found for digit detection.")
    detected_digits = []

    for contour in contours:
        # Filter out small noise artifacts by checking bounding area
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 20 and w<= 50 and h >= 30:
            # Extract the region of interest (ROI) for the digit
            roi = thresh[y:y+h, x:x+w]
            
            # 4. Resize ROI to match your classifier's input structure (e.g., 20x20)
            roi_resized = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
            
            # Draw bounding boxes on the original image for visualization
            cv2.rectangle(gray, (x, y), (x + w, y + h), 0, 2) # fix this for gray image
            
            # Store for classification phase
            detected_digits.append(roi_resized)
    
    return gray, contours, detected_digits

def split_into_panels(img, n=7, crop_top=0.02, crop_bottom=0.05, crop_side=0.01):
    h, w = img.shape[:2]
    panel_w = w / n
    panels = []

    for i in range(n):
        x1 = int(i * panel_w)
        x2 = int((i + 1) * panel_w)

        # Optional margin crop to remove borders/glare edges
        y1 = int(h * crop_top)
        y2 = int(h * (1.0 - crop_bottom))
        xx1 = x1 + int((x2 - x1) * crop_side)
        xx2 = x2 - int((x2 - x1) * crop_side)

        panels.append(img[y1:y2, xx1:xx2])
    return panels

def get_closed_image(binary_img):

    # Morphological closing to fill gaps in segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_image = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed_image

def preprocess_for_lcd(panel):
    # LCD has colored background and dark text.
    # Use value channel + local contrast to isolate characters.
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    # Improve local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    v_eq = clahe.apply(v)

    # Blur small noise
    blur = cv2.GaussianBlur(v_eq, (5, 5), sigmaX=2, sigmaY=2)

    # Dark text on bright background => invert binary threshold
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )

    canny_edges, sobel_edges = detect_edges(blur)

    # dig, contours, detected_digits = detect_digits(v)

    # Morph cleanup
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    closed_img = get_closed_image(thr)
    
    # sobel_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_OPEN, kernel, iterations=1)
    # sobel_edges = cv2.morphologyEx(sobel_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    # sobel_edges = get_closed_image(sobel_edges)

    return thr, closed_img, v, v_eq, blur, canny_edges, sobel_edges


def crop_middle_grid(panel, horizontal_parts=5, vertical_parts=4, keep_h=(1, 2, 3), keep_v=(1, 2)):
    """Keep center cells from an NxM grid split of the panel."""
    h, w = panel.shape[:2]
    ys = np.linspace(0, h, horizontal_parts + 1, dtype=int)
    xs = np.linspace(0, w, vertical_parts + 1, dtype=int)

    y1 = ys[min(keep_h)]
    y2 = ys[max(keep_h) + 1]
    x1 = xs[min(keep_v)]
    x2 = xs[max(keep_v) + 1]

    return panel[y1:y2, x1:x2]


def hconcat_with_uniform_height(images, target_height, interpolation=cv2.INTER_AREA):
    resized = []
    for image in images:
        height, width = image.shape[:2]
        if height != target_height:
            new_width = max(int(width * (target_height / height)), 1)
            image = cv2.resize(image, (new_width, target_height), interpolation=interpolation)
        resized.append(image)
    return cv2.hconcat(resized)


def build_debug_row(images, output_width, gray=False, interpolation=cv2.INTER_NEAREST):
    if gray:
        images = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in images]

    row_height = min(image.shape[0] for image in images)
    row = hconcat_with_uniform_height(images, row_height, interpolation=interpolation)
    row = cv2.resize(row, (output_width, row.shape[0]), interpolation=interpolation)
    return row


def build_single_debug_image(original_img, middle_crops, bin_images, closed_images, value_images, value_eq_images, blur_images, canny_images, sobel_images):
    output_width = original_img.shape[1]
    row_specs = [
        (middle_crops, False, cv2.INTER_AREA),
        (value_images, True, cv2.INTER_NEAREST),
        (value_eq_images, True, cv2.INTER_NEAREST),
        (blur_images, True, cv2.INTER_NEAREST),
        (canny_images, True, cv2.INTER_NEAREST),
        (sobel_images, True, cv2.INTER_NEAREST),
        (bin_images, True, cv2.INTER_NEAREST),
        (closed_images, True, cv2.INTER_NEAREST)
    ]

    stacked_rows = [original_img]
    for images, gray, interpolation in row_specs:
        stacked_rows.append(build_debug_row(images, output_width, gray=gray, interpolation=interpolation))

    return cv2.vconcat(stacked_rows)


def ocr_panel(binary_img):
    # Numeric-only OCR for inverter values.
    configs = [
        '--oem 3 --psm 7 -c tessedit_char_whitelist=Av%0123456789.- -c classify_bln_numeric_mode=1',
        '--oem 3 --psm 13 -c tessedit_char_whitelist=Av%0123456789.- -c classify_bln_numeric_mode=1',
    ]

    best_text = ""
    best_score = -1
    string = ""
    for cfg in configs:
        # data = pytesseract.image_to_data(
        #     binary_img, config=cfg, output_type=pytesseract.Output.DICT
        # )
        string = string + " " + pytesseract.image_to_string(binary_img, config=cfg).strip()

        words = []
        confs = []
        # for txt, conf in zip(data["text"], data["conf"]):
        #     txt = txt.strip()
        #     try:
        #         c = float(conf)
        #     except ValueError:
        #         c = -1
        #     if txt and c >= 0:
        #         words.append(txt)
        #         confs.append(c)

        # text = " ".join(words).strip()
        # number_tokens = re.findall(r"-?\d+(?:\.\d+)?", text)
        # text = number_tokens[0] if number_tokens else ""
        # score = np.mean(confs) if confs else -1

        # if score > best_score:
        #     best_score = score
        #     best_text = text

    # Light normalization
    # best_text = re.sub(r"\s+", " ", best_text).strip()
    return string.strip(), best_score


def read_inverter_screens(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    panels = split_into_panels(img, n=7)
    rows = []
    middle_crops = []
    bin_images = []
    closed_images = []
    value_images = []
    value_eq_images = []
    blur_images = []
    canny_images = []
    sobel_images = []
    if debug:
        os.makedirs("debug", exist_ok=True)

    for idx, panel in enumerate(panels, start=1):
        middle_crop = crop_middle_grid(
            panel,
            horizontal_parts=4,
            vertical_parts=5,
            keep_h=(1, 2),
            keep_v=(1, 2, 3)
        )

        bin_img, closed_img, v, v_eq, blur, canny_edges, sobel_edges = preprocess_for_lcd(middle_crop)

        panel_row = {"panel": idx}
        names = ["binary", "closed", "value", "value_eq", "blur", "canny", "sobel"]
        images = [bin_img, closed_img, v, v_eq, blur, canny_edges, sobel_edges]
        for name, proc_img in zip(names, images):
            text, score = ocr_panel(proc_img)
            panel_row[f"{name}_text"] = text
            # panel_row[f"{name}_score"] = round(float(score), 2)

        rows.append(panel_row)
        middle_crops.append(middle_crop)
        bin_images.append(bin_img)
        closed_images.append(closed_img)
        value_images.append(v)
        value_eq_images.append(v_eq)
        blur_images.append(blur)
        canny_images.append(canny_edges)
        sobel_images.append(sobel_edges)
    if debug:
        debug_stack = build_single_debug_image(img, middle_crops, bin_images, closed_images, value_images, value_eq_images, blur_images, canny_images, sobel_images)
        cv2.imwrite("debug_stacked2.jpg", debug_stack)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    image_path = "detected_rectangles.jpg"  # change to your file
    out = read_inverter_screens(image_path, debug=True)

    print("OCR results:")
    print(out.to_string(index=False))