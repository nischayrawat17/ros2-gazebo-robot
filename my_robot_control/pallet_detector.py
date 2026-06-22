import cv2
import numpy as np

def detect_pallet(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Tan/cardboard color range in HSV
    lower_tan = np.array([10, 60, 100])
    upper_tan = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_tan, upper_tan)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No pallet/box detected")
        return

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    center_x = x + w // 2
    center_y = y + h // 2

    print(f"Detected box at pixel ({center_x}, {center_y}), size {w}x{h}")

    output = img.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(output, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.imwrite('/tmp/detection_result.png', output)
    print("Saved annotated result to /tmp/detection_result.png")

if __name__ == '__main__':
    detect_pallet('pallet_test.png')
