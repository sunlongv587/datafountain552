from ultralytics import YOLO

# Load a model
model = YOLO(model='yolov8.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO(model='./dataset/df.yaml', task='detect').load('./models/yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='./dataset.yaml', epochs=100, batch=8, imgsz=2600)