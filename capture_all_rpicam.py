import glob
import os
import re
import subprocess
import cv2

CONTOUR_OFFSET = 10
CAPTURE_CMD = "rpicam-still -t 32000 --timelapse 4000 --roi 0.5,0.5,0.2,0.2 -o image%d.jpg"


def natural_key(path: str):
    # Sort image2 before image10
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
    return edged, gray, blurred


def get_contour(edged):
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
            contour_area = cv2.contourArea(approx)

            if rect_area > 0 and abs(1.0 - (contour_area / rect_area)) < 0.15:
                right_contours.append(contour)

    print(f"Found {len(right_contours)} right-angled rectangles.")
    if right_contours:
        return max(right_contours, key=cv2.contourArea)
    return None


def crop_with_contour(img_bgr, contour):
    x, y, w, h = cv2.boundingRect(contour)

    y1 = max(y + CONTOUR_OFFSET, 0)
    y2 = min(y + h - CONTOUR_OFFSET, img_bgr.shape[0])
    x1 = max(x + CONTOUR_OFFSET, 0)
    x2 = min(x + w - CONTOUR_OFFSET, img_bgr.shape[1])

    if y1 >= y2 or x1 >= x2:
        raise ValueError("Invalid crop bounds. Adjust CONTOUR_OFFSET or contour detection.")

    return img_bgr[y1:y2, x1:x2]


def capture_all_from_files(pattern="image*.jpg"):
    image_files = sorted(glob.glob(pattern), key=natural_key)
    if not image_files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    first_img = cv2.imread(image_files[0])
    if first_img is None:
        raise ValueError(f"Could not read image: {image_files[0]}")

    edged, _, _ = get_edged_from_image(first_img)
    largest_contour = get_contour(edged)
    if largest_contour is None:
        cv2.imwrite("error.jpg", first_img)
        raise RuntimeError("No right-angled rectangles found in first image.")

    crops = []
    for idx, file_path in enumerate(image_files):
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping unreadable image: {file_path}")
            continue

        try:
            crops.append(crop_with_contour(img, largest_contour))
        except Exception as e:
            print(f"Error processing {file_path} (index {idx}): {e}")

    if not crops:
        raise RuntimeError("No cropped images produced.")

    return cv2.hconcat(crops)


def save_all(output_file="detected_rectangles.jpg"):
    run_capture_command()
    result = capture_all_from_files("image*.jpg")
    cv2.imwrite(output_file, result)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    save_all("detected_rectangles.jpg")