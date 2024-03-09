import multiprocessing
import cv2
import numpy as np
import time
import math
import os
from collections import deque
import glob
import tensorflow as tf
from ultralytics import YOLO
import supervision as sv



video_path=2
# video_path='main__.mp4'
# video_path="D:/New folder/Dataset/Stair/reversef_012.avi"
# video_path="D:/New folder/Dataset/Walk/reverser_012.avi"




SEQUENCE_LENGTH=30
model_new = tf.keras.models.load_model("C:/Video Classification/downloaded_model2.h5")
box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )


def capture(frame_queue, initial_frame_dict, bbox_mask, bbox_obj, poseD_):
    pose = ''
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH*2))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT*2))
    
    result = cv2.VideoWriter('new.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    
    ret, initial_frame = cap.read()

    if not ret:
        print("Error capturing initial frame")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    time.sleep(2)
    initial_frame_dict['frame'] = initial_frame  # Store the initial frame in the shared dictionary

    x, y, w, h = 0, 0, 0, 0
    for result in model.track(video_path, stream=True, agnostic_nms=True):
            frame = result.orig_img
            frame_queue.put(frame)
            detections = sv.Detections.from_ultralytics(result)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            if 'pose' in poseD_:
                pose = poseD_['pose']

            labels = [
                f"{tracker_id} {model.model.names[class_id]} {confidence:.2f}" if class_id != 0
                else f"{tracker_id} Person {confidence:.2f}"
                for (box, _, confidence, class_id, tracker_id) in detections
            ]
            for (box, _, confidence, class_id, tracker_id) in detections:
                if class_id != 0:
                    text=''
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)

                else:
                    text=f'Activity: {pose}'
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 2)
            


            frame = box_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            # Handle user input to close the window
            cv2.imshow("Video Feed", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



#def mask(frame_queue, initial_frame_dict, bbox_mask):
#     while 'frame' not in initial_frame_dict:
#         time.sleep(0.1)

#     while True:
#         frame = frame_queue.get()
#         initial_frame = initial_frame_dict['frame']  # Retrieve the initial frame from the shared dictionary

#         frame_diff = cv2.absdiff(initial_frame, frame)
#         gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

#         _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for contour in contours:
#             if cv2.contourArea(contour) > 1000:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 bbox_mask['x'], bbox_mask['y'], bbox_mask['w'], bbox_mask['h'] = x, y, w, h






         

            






def poseD(frame_queue,poseD_):
    IMAGE_HEIGHT = 64
    IMAGE_WIDTH = 64
    CLASSES_LIST = ['***SAFE***', '/*/*SUSPICIOUS/*/*']


    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    text = ''

    # Iterate until the video is accessed successfully.
    while True:
        # if()
        print("Hi person")
        # Read the frame.
        frame = frame_queue.get()
        #time.sleep(1/10)


        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list.
        frames_queue.append(normalized_frame)
        #frames_queue.append(normalized_frame)

        # Check if the number of frames in the queue are equal to the fixed sequence length.
        if len(frames_queue) == SEQUENCE_LENGTH:
           
            # Pass the normalized frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model_new.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]
            probability = str(round(predicted_labels_probabilities[predicted_label] * 100, 2))
            text= f"{predicted_class_name} {probability}"

        poseD_['pose']=text
        



        




if __name__ == "__main__":
    frame_queue = multiprocessing.Queue()
    
    manager = multiprocessing.Manager()
    initial_frame_dict = manager.dict()
    bbox_mask = manager.dict()
    bbox_obj = manager.dict()
    poseD_ = manager.dict()
    
    
    
    
    
    process1 = multiprocessing.Process(target=capture, args=(frame_queue, initial_frame_dict, bbox_mask,bbox_obj,poseD_))
    #process2 = multiprocessing.Process(target=mask, args=(frame_queue, initial_frame_dict, bbox_mask))
  
    process4 = multiprocessing.Process(target=poseD, args=(frame_queue, poseD_))
    

    process1.start()
    #process2.start()

    process4.start()
    

    process1.join()
    #process2.join()

    process4.join()

