# lena
import cv2
import os
from ultralytics import YOLO

# Load the YOLOv8 model n=微小的 pt = pytorch
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO(model='./weights/best.pt')


# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category


def image_detect():
    test_dir = os.path.expanduser("~/Notes/Datasets/person_and_car_dataset_from_datafountain/test_images")
    img_list = os.listdir(test_dir)
    for img in img_list:
        img_path = os.path.join(test_dir, img)
        # Predict with the model
        results = model(source=img_path, show=False, save=True)  # predict on an image
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        key = cv2.waitKey(0)
        if key == 113:
            break
    cv2.destroyAllWindows()


# Open the video file
# video_path = "path/to/your/video/file.mp4"
def video_detect():
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        res, frame = cap.read()

        if res:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # video_detect()
    image_detect()
