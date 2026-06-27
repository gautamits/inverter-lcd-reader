import glob
import os
import re
import subprocess
import cv2
import numpy as np

# Keep CPU load low on Pi Zero
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

CONTOUR_OFFSET = 6
SCALE = 1  # process at half-size to reduce CPU and heat

# Lower-res, no preview, lower quality, 4s timelapse for ~32s
CAPTURE_CMD = (
    "rpicam-still -n -t 32000 --timelapse 4000 "
    "--width 1280 --height 720 --quality 100 "
    "--roi 0.5,0.5,0.5,0.5 -o image%d.jpg"
)


def natural_key(path: str):
    name = os.path.basename(path)
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def run_capture_command():
    print("Running camera capture command...")
    subprocess.run(["bash", "-lc", CAPTURE_CMD], check=True)


def get_edged_from_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged


def get_contour(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours = []

    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            contour_area = cv2.contourArea(approx)
            if rect_area > 0 and abs(1.0 - (contour_area / rect_area)) < 0.18:
                right_contours.append(contour)

    print(f"Found {len(right_contours)} right-angled rectangles.")
    return max(right_contours, key=cv2.contourArea) if right_contours else None


def crop_with_contour(img_bgr, contour):
    x, y, w, h = cv2.boundingRect(contour)

    y1 = max(y + CONTOUR_OFFSET, 0)
    y2 = min(y + h - CONTOUR_OFFSET, img_bgr.shape[0])
    x1 = max(x + CONTOUR_OFFSET, 0)
    x2 = min(x + w - CONTOUR_OFFSET, img_bgr.shape[1])

    if y1 >= y2 or x1 >= x2:
        raise ValueError("Invalid crop bounds. Adjust contour settings.")

    return img_bgr[y1:y2, x1:x2]


def read_scaled(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if SCALE != 1.0:
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return img


def capture_all_from_files(pattern="image*.jpg"):
    image_files = sorted(glob.glob(pattern), key=natural_key)
    if not image_files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    first_img = read_scaled(image_files[0])
    if first_img is None:
        raise ValueError(f"Could not read image: {image_files[0]}")

    contour = get_contour(get_edged_from_image(first_img))
    if contour is None:
        cv2.imwrite("error.jpg", first_img)
        raise RuntimeError("No right-angled rectangles found in first image.")

    # First valid crop defines output geometry
    first_crop = crop_with_contour(first_img, contour)
    h, w, c = first_crop.shape

    # Preallocate final strip to avoid repeated hconcat allocations
    result = np.zeros((h, w * len(image_files), c), dtype=first_crop.dtype)

    write_idx = 0
    for idx, file_path in enumerate(image_files):
        img = read_scaled(file_path)
        if img is None:
            print(f"Skipping unreadable image: {file_path}")
            continue
        try:
            crop = crop_with_contour(img, contour)
            if crop.shape[:2] != (h, w):
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            result[:, write_idx * w:(write_idx + 1) * w] = crop
            write_idx += 1
        except Exception as e:
            print(f"Error processing {file_path} (index {idx}): {e}")

    if write_idx == 0:
        raise RuntimeError("No cropped images produced.")

    # Trim if some images failed
    return result[:, :write_idx * w]


def save_all(output_file="detected_rectangles.jpg"):
    run_capture_command()
    result = capture_all_from_files("image*.jpg")
    cv2.imwrite(output_file, result, [cv2.IMWRITE_JPEG_QUALITY, 70])
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    save_all("detected_rectangles.jpg")