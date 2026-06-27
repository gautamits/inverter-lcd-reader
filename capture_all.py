import time
import cv2
from picamera2 import Picamera2


WIDTH=1280
HEIGHT=720
CONTOUR_OFFSET=10

print("Camera warming up...")
time.sleep(2) 

picam = Picamera2()
picam.configure(picam.create_still_configuration(main={"format": "RGB888", "size": (WIDTH, HEIGHT)}))
picam.start()

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

def capture_cropped(contour):
    edged, gray, blurred, img_bgr = get_edged()
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = img_bgr[y+CONTOUR_OFFSET:y+h-CONTOUR_OFFSET, x+CONTOUR_OFFSET:x+w-CONTOUR_OFFSET]
    return cropped_image


def capture_all(img_count, time_gap):
    edged, gray, blurred, img_bgr = get_edged()
    largest_contour = get_contour(edged)
    if largest_contour is None:
        print("No right-angled rectangles found.")
        cv2.imwrite('error.jpg', img_bgr)
        exit()
    cropped=capture_cropped(largest_contour)
    result=cv2.hconcat([cropped])
    time.sleep(time_gap)
    for index, value in enumerate(range(img_count), start=0):
        try:
            cropped = capture_cropped(largest_contour)
            result=cv2.hconcat([result, cropped])
        except Exception as E:
            print(f"Error capturing image {index}: {E}")
            pass
        time.sleep(time_gap)
    picam.stop()
    return result

def save_all(file_name, img_count, time_gap):
    result=capture_all(img_count, time_gap)
    cv2.imwrite(file_name, result)

if __name__ == "__main__":
    save_all("detected_rectangles.jpg", 7, 3)