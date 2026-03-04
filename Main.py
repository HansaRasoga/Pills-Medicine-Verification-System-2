import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ----------------------------
# Dataset Path
# ----------------------------
DATASET_PATH = r"C:\Users\Hansa\Music\Sem 5\ImageP\archive\ePillID_data\classification_data\segmented_nih_pills_224"
OUTPUT_FILE = "results.csv"

# ----------------------------
# Shape Detection Function
# ----------------------------
def detect_shape(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1000:

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            vertices = len(approx)

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            circularity = (4 * np.pi * area) / (peri * peri)

            if vertices == 3:
                return "Triangle"

            elif vertices == 4:
                if 0.9 <= aspect_ratio <= 1.1:
                    return "Square"
                else:
                    return "Rectangle"

            elif circularity > 0.8:
                return "Round"

            elif 0.5 < circularity <= 0.8:
                return "Oval / Elliptical"

            else:
                return "Capsule-shaped (Oblong)"

    return "Unknown"


# ----------------------------
# Color Detection Function
# ----------------------------
def detect_color(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "White": ([0,0,200], [180,30,255]),
        "Yellow": ([20,100,100], [35,255,255]),
        "Blue": ([100,150,0], [140,255,255]),
        "Red": ([0,100,100], [10,255,255]),
        "Green": ([40,70,70], [80,255,255]),
        "Purple": ([130,50,50], [160,255,255]),
        "Orange": ([10,100,100], [20,255,255]),
        "Brown": ([10,100,20], [20,255,200]),
        "Black": ([0,0,0], [180,255,30])
    }

    detected = []

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 500:
            detected.append(color)

    if len(detected) == 0:
        return "Unknown"

    return " & ".join(detected)


# ----------------------------
# Process Entire Dataset
# ----------------------------
results = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith((".jpg", ".png", ".jpeg")):

            path = os.path.join(root, file)
            image = cv2.imread(path)

            if image is None:
                continue

            image = cv2.resize(image, (500, 500))

            shape = detect_shape(image)
            color = detect_color(image)

            print(f"{file} → Shape: {shape}, Color: {color}")

            results.append([file, shape, color])


# ----------------------------
# Save Results to CSV
# ----------------------------
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image Name", "Detected Shape", "Detected Color"])
    writer.writerows(results)

print("\nProcessing Complete.")
print("Results saved to results.csv")
