import cv2
from  ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

drawing = False
line_start = None
line_end = None
line_drawn = False 

crossed=set()
person_count=0

def draw_line(event,x,y, flags,param):
    global drawing, line_start, line_end,line_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_start = (x,y)
        line_end = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            line_end = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        line_end = (x,y)
        drawing = False
        line_drawn = True

def check_cross(cur_center, prev_center):
    global line_start,line_end
    def cross(ax,ay,bx,by,cx,cy):
        return (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    
    x1,y1 =cur_center
    x2,y2=prev_center
    x3,y3=line_start
    x4,y4=line_end

    d1 = cross(x1,y1,x2,y2,x3,y3)
    d2 = cross(x1,y1,x2,y2,x4,y4)
    d3 = cross(x3,y3,x4,y4,x1,y1)
    d4 = cross(x3,y3,x4,y4,x2,y2)

    if d1==0 and d2 ==0:
        if max(x1,x2)<min(x3,x4) or min(x1,x2)>max(x3,x4):
            return False
        if max(y1,y2)<min(y3,y4) or min(y1,y2)>max(y3,y4):
            return False
        return True 
    return (d1*d2<=0)and(d3*d4<=0)

cv2.namedWindow('draw a line')
cv2.setMouseCallback('draw a line', draw_line)

model = YOLO("yolo11n.pt")
tracker = DeepSort(max_age = 50, n_init=2, nms_max_overlap=1.0)

cap = cv2.VideoCapture('video2.mov')

ret, frame = cap.read()

if not ret:
    exit()
while True:
    clear_frame=frame.copy()
    if line_start and line_end:
        cv2.line(clear_frame,line_start,line_end,(0,0,255),2)
    cv2.imshow('draw a line', clear_frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        if line_drawn and line_start!=line_end:
            break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow('draw a line')

prev_centers={}
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter +=1
    detections = model(frame)[0]
    results = []

    for data in detections.boxes.data.tolist():
        conf = float(data[4])
        if conf < 0.3:
            continue
        x_min, y_min, x_max, y_max = int(data[0]),int(data[1]),int(data[2]),int(data[3])
        class_id = data[5]

        if class_id == 0:
            results.append([[x_min,y_min,x_max-x_min,y_max-y_min],conf,class_id])
    
    tracks = tracker.update_tracks(results,frame = frame)

    cv2.line(frame, line_start,line_end,(0,0,255),2)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        x_min, y_min, x_max, y_max = int(ltrb[0]),int(ltrb[1]), int(ltrb[2]),int(ltrb[3])
        cur_center = ((x_min+x_max)//2,(y_min+y_max)//2)

        if frame_counter%10 ==0:
            if track_id in prev_centers and track_id not in crossed:
                if check_cross(cur_center, prev_centers[track_id]):
                    crossed.add(track_id)
                    person_count+=1

            prev_centers[track_id] = cur_center 

        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max), (0,255,0),2)
        cv2.rectangle(frame,(x_min,y_min-20),(x_min+60,y_min),(0,255,0),-1)
        cv2.putText(frame, f"ID: {track_id}", (x_min+5,y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        cv2.circle(frame, cur_center,3,(255,0,0),-1)

    cv2.putText(frame, f"Counter: {person_count}",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.imshow('frame',cv2.resize(frame,(900,600)))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()