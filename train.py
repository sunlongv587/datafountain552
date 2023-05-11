from ultralytics import YOLO
import os



# Load a model
# model = YOLO(model='yolov8.yaml')  # build a new model from YAML
# 加载上次训练的模型，继续训练 填写上次训练的 /content/drive/MyDrive/datafountain-552/workspace/datafountain552/runs/detect/train7/weight/last.pt

last_model_path = '/content/drive/MyDrive/datafountain-552/workspace/datafountain552/runs/detect/train0002/weights/last.pt'
print(f"上次训练模型 last.pt 路径是：{last_model_path}")
model = YOLO(model=last_model_path)

# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO(model='./dataset/df.yaml', task='detect').load('./models/yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='./dataset.yaml', epochs=100, batch=4, imgsz=2600)


