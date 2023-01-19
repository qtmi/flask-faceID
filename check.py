import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set the video frame width and height
cap.set(3, 640)
cap.set(4, 480)

# Loop to capture video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if a frame was read correctly
    if not ret:
        break

    # Display the video frame
    cv2.imshow("Webcam", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
