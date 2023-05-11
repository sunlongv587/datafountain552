from ultralytics import YOLO
import os



# Load a model
# model = YOLO(model='yolov8.yaml')  # build a new model from YAML
# 加载上次训练的模型，继续训练 填写上次训练的 /content/drive/MyDrive/datafountain-552/workspace/datafountain552/runs/detect/train7/weight/last.pt

def find_last_model(runs_path):
    # train_dir = '/content/drive/MyDrive/datafountain-552/workspace/datafountain552/runs/detect/train0002/weights/last.pt'
    detect_dir = 'detect'
    detect_path = os.path.join(runs_path, detect_dir)
    folders = []
    # 读取detect目录下所有文件夹信息
    for entry in os.scandir(detect_path):
        if entry.is_dir():
            folders.append(entry)

    # 按创建时间倒序排列所有文件夹
    sorted_folders = sorted(folders, key=lambda e: e.stat().st_ctime, reverse=True)

    # 打印所有文件夹名
    for folder in sorted_folders:
        last_model_path = os.path.join(detect_path, folder.name, 'weights', 'last.pt')
        if os.path.isfile(last_model_path):
            print(f"上次训练的文件找到，位置：{last_model_path}")
            return last_model_path
        else:
            print(f"文件不存在：{last_model_path}, 继续查找上次训练的模型")


def main():
    # last_model_path = '/content/drive/MyDrive/datafountain-552/workspace/datafountain552/runs/detect/train0002/weights/last.pt'
    last_model_path = find_last_model("/content/drive/MyDrive/datafountain-552/workspace/datafountain552/runs")
    if last_model_path == None:
        last_model_path = "yolov8n.yaml"
    print(f"加载上次训练的模型，last.pt 路径是：{last_model_path}")
    model = YOLO(model=last_model_path)

    # Train the model
    model.train(data='./dataset.yaml', epochs=100, batch=4, imgsz=2600)

if __name__ == '__main__':
    main()

