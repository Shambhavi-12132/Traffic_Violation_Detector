import cv2
from ultralytics import YOLO

# Global variables to store line coordinates and violators
line_points = []
violators = set()
violation_count = 0

# Mouse callback function to get points for the line
def get_mouse_clicks(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print(f"Point {len(line_points)} added: ({x}, {y})")

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Use live camera feed
cap = cv2.VideoCapture(0) 
window_name = "Live Red Light Detector"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, get_mouse_clicks)

print("Please click on two points on the screen to draw the stop line, then press 's' to start detection.")

# First, capture frames just to let the user draw the line
while len(line_points) < 2:
    success, frame = cap.read()
    if not success:
        break
    
    # Draw points on the frame as they are clicked
    for point in line_points:
        cv2.circle(frame, point, 5, (0, 255, 255), -1)
    
    cv2.putText(frame, "Click 2 points for the stop line, then press 's'", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

# --- Main detection loop ---
while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Assume light is RED for now. Add logic here to check light status.
    TRAFFIC_LIGHT_IS_RED = True

    if TRAFFIC_LIGHT_IS_RED and len(line_points) == 2:
        # Draw the line
        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                bottom_center_y = int(y2)
                
                # Simple check: Does the bottom of the box cross the line's average Y coordinate?
                # This is a simplification; a more robust check would use line intersection geometry.
                line_y_avg = (line_points[0][1] + line_points[1][1]) / 2
                
                if bottom_center_y > line_y_avg and track_id not in violators:
                    violators.add(track_id)
                    violation_count += 1
    
    # Display violation count
    cv2.putText(frame, f"Violations: {violation_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()






# import cv2   
# from ultralytics import YOLO 

# # Load the YOLOv8 model
# # 'yolov8n.pt' is the nano version, fast but less accurate.
# # 'yolov8s.pt' (small) or 'yolov8m.pt' (medium) are good balances.
# model = YOLO('yolov8s.pt')

# # Open the video file
# #video_path = "mixkit-times-square-during-a-sunny-day-4442-hd-ready.mp4" # <--- IMPORTANT: Change this to your video file
# cap = cv2.VideoCapture(0)

# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         # The 'persist=True' argument tells the tracker that the current image or frame is the next in a sequence.
#         results = model.track(frame, persist=True, classes=[2, 3, 5, 7]) # classes for car, motorcycle, bus, truck

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()