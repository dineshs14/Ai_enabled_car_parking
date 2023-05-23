import cv2
import pickle
import cvzone
import numpy as np

# Define the width and height of ROI
width, height = 107, 48

# Loading the ROI from parkingSlotPosition file
try:
    with open('Model building/parkingSlotPosition', 'rb') as f:
        posList = pickle.load(f)
        print("parkingSlotPosition file loaded successfully.")
except:
    print("Failed to load parkingSlotPosition file.")
    posList = []

# Read the video
cap = cv2.VideoCapture('F:\naan mudhalvan\ai enabled car parking\Data\carParkingInput.mp4')

while True:
    # Looping the video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Reading frame by frame from video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    # Apply median blur
    median = cv2.medianBlur(threshold, 5)

    # Dilate the image
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)

    # Create a copy of the original frame to overlay the text
    imgOverlay = frame.copy()

    # Initialize lists to store free slot numbers and positions
    freeSlots = []
    freeSlotPositions = []

    # Check parking space and display slot numbers
    for i, pos in enumerate(posList):
        x, y = pos
        imgCrop = dilated[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            color = (0, 255, 0)
            thickness = 2
            freeSlots.append(i + 1)
            freeSlotPositions.append(pos)
        else:
            color = (0, 0, 255)
            thickness = 2
        cv2.rectangle(imgOverlay, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cv2.putText(imgOverlay, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness,
                    cv2.LINE_AA)

    # Display the number of free parking slots
    cv2.putText(imgOverlay, f'Free: {len(freeSlots)} / {len(posList)}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)

    # Display the free slot positions
    for slot, pos in zip(freeSlots, freeSlotPositions):
        cv2.putText(imgOverlay, f'Slot {slot}', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    thickness, cv2.LINE_AA)

    # Blend the overlay image with the original frame
    imgOutput = cv2.addWeighted(frame, 0.7, imgOverlay, 0.3, 0)

    # Display the frame
    cv2.imshow("Car Parking Input", imgOutput)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Read the video
cap = cv2.VideoCapture(r'F:\naan mudhalvan\ai enabled car parking\Data\carParkingInput.mp4')

while True:
    # Looping the video
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Reading frame by frame from video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

    # Apply median blur
    median = cv2.medianBlur(threshold, 5)

    # Dilate the image
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)

    # Initialize lists to store free slot numbers and positions
    freeSlots = []
    freeSlotPositions = []

    # Check parking space and display slot numbers
    for i, pos in enumerate(posList):
        x, y = pos
        imgCrop = dilated[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 900:
            color = (0, 255, 0)
            freeSlots.append(i + 1)
            freeSlotPositions.append(pos)
            cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, 2)
            cv2.putText(frame, f'Slot {i + 1}', (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        cv2.LINE_AA)
        else:
            color = (0, 0, 255)
            cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, 2)
            cv2.putText(frame, f'Slot {i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Display the available parking slot count / total parking slot count
    cvzone.putTextRect(frame, f'Free: {len(freeSlots)} / {len(posList)}', (10, 25), scale=1.5,
                       thickness=2, offset=5, colorR=(0, 200, 0))

    # Display the slot numbers for free slots
    if freeSlots:
        slot_text = 'Free Slots: ' + ', '.join(map(str, freeSlots))
        cvzone.putTextRect(frame, slot_text, (15, 50), scale=1.5, thickness=2, offset=3, colorR=(0, 200, 0))

    # Display the frame
    cv2.imshow("Car Parking Input", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()