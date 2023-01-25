import numpy as np
import cv2

# Start the webcam
cap = cv2.VideoCapture(0)
points = []
def mouse_callback(event, x,y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [x,y]
        points.append(coords)


cv2.namedWindow('Webcam')
cv2.setMouseCallback("Webcam", mouse_callback)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Define the vertices of the polygon

    pts = np.array(points, np.int32)

    # Reshape the vertices to a 2xN array
    pts = pts.reshape((-1,1,2))

    # Draw the polygon on the webcam screen
    cv2.polylines(frame,[pts],True,(0,0,255),3)

    # Display the webcam screen
    cv2.imshow("Webcam", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
