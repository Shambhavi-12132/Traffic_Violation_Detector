import cv2
from ultralytics import YOLO
import numpy as np

# --- Global variables ---
line_points = []
violators = set()
violation_count = 0
light_status = "GREEN"  # Start with the light as GREEN

# --- UI Drawing Function ---
def draw_ui(frame, light_status, violation_count):
    """
    This function draws all the UI elements on top of the video frame.
    """
    # --- Step 1: Define the colors we will use ---
    RED_ON = (0, 0, 255)       # Bright red for when the light is on
    RED_OFF = (30, 30, 100)    # Dim red for when the light is off
    GREEN_ON = (0, 255, 0)     # Bright green
    GREEN_OFF = (30, 100, 30)  # Dim green
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    
    # Get the height and width of the video frame
    h, w, _ = frame.shape

    # --- Step 2: Draw the Traffic Light background ---
    # We draw a black rectangle in the top-right corner (w-110, 10)
    cv2.rectangle(frame, (w - 110, 10), (w - 10, 150), BLACK, -1)
    
    # --- Step 3: Decide which light color is bright and which is dim ---
    if light_status == "RED":
        red_color = RED_ON      # If status is RED, use bright red
        green_color = GREEN_OFF # and dim green.
    else: # (light_status == "GREEN")
        red_color = RED_OFF     # If status is GREEN, use dim red
        green_color = GREEN_ON  # and bright green.

    # --- Step 4: Draw the two circles for the lights ---
    # Draw the red light circle. Its color is decided above.
    cv2.circle(frame, (w - 60, 45), 25, red_color, -1)
    # Draw the green light circle below it.
    cv2.circle(frame, (w - 60, 105), 25, green_color, -1)

    # --- Step 5: Draw the text panel on the left ---
    # This part draws the violation counter and instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0) # Make it semi-transparent
    
    cv2.putText(frame, "RED LIGHT VIOLATION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(frame, f"VIOLATIONS: {violation_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, RED_ON, 2)
    cv2.putText(frame, "Press 'R' for Red, 'G' for Green", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    return frame

# --- Mouse callback and main program logic (This part remains the same as before) ---

def get_mouse_clicks(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(0)
window_name = "Live Red Light Detector"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, get_mouse_clicks)

print("Please click two points to draw the stop line, then press 'S' to start.")

# Loop to set the line
while True:
    success, frame = cap.read()
    if not success: break
    
    for point in line_points:
        cv2.circle(frame, point, 5, (0, 255, 255), -1)
    if len(line_points) == 2:
        cv2.line(frame, line_points[0], line_points[1], (0, 255, 255), 2)
        
    cv2.putText(frame, "Click 2 points for the line, then press 'S'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s") and len(line_points) == 2:
        break
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Main detection loop
while True:
    success, frame = cap.read()
    if not success: break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        light_status = "RED"
        violators.clear()
    elif key == ord('g'):
        light_status = "GREEN"
        violators.clear()
    elif key == ord('q'):
        break

    if light_status == "RED":
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                bottom_center_y = int(y2)
                line_y_avg = (line_points[0][1] + line_points[1][1]) / 2
                
                if bottom_center_y > line_y_avg and track_id not in violators:
                    violators.add(track_id)
                    violation_count += 1
    
    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)
    
    frame_with_ui = draw_ui(frame, light_status, violation_count)
    cv2.imshow(window_name, frame_with_ui)

cap.release()
cv2.destroyAllWindows()