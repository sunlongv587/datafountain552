import os
import json

# source dir
# df_dir = os.path.expanduser(r"~/Datasets/person_and_car_dataset_from_datafountain/train_dataset")
df_dir = "/content/drive/MyDrive/dataset/train_dataset"
file_name = "train.json"
max_width = 1280
max_height = 720
label_dict = {'<UNK>': -1}

# target dir
# yolo_dir = os.path.expanduser(r"~/Datasets/person_and_car_dataset_from_datafountain/train_dataset/labels")
yolo_dir = "/content/drive/MyDrive/dataset/train_dataset/labels"
file_suffix = ".txt"

file_stream_cache = {}


def get_file_stream(file_name):
    os.makedirs(yolo_dir, exist_ok=True)
    file_stream = file_stream_cache.get(file_name)
    if file_stream is None:
        file_stream = open(os.path.join(yolo_dir, f"{file_name.split('.')[0]}{file_suffix}"), "a")
        file_stream_cache[file_name] = file_stream
    return file_stream


def close_all_file_stream():
    for file_stream in file_stream_cache.values():
        file_stream.close()


def main():
    with open(os.path.join(df_dir, file_name), 'r') as f:
        content = f.read().replace('\n', '').replace(" ", "")
        content_json = json.loads(content)
        annotations = content_json['annotations']
        for annotation in annotations:
            # 解析文件名，相同的文件名写到同一个文件里
            filename = annotation['filename']
            filename = filename.split('\\')[1]
            # 记录所有的label,为label做one-hot编码
            label = annotation['label']
            cls = label_dict.get(label)
            if cls is None:
                cls = len(label_dict) - 1
                label_dict[label] = cls
            # 解析 box
            box = annotation['box']
            if box['xmin'] is None or box['ymin'] is None or box['xmax'] is None or box['ymax'] is None:
                print(f"error annotation, {annotation}")
                continue
            xmin = float(box['xmin'])
            ymin = float(box['ymin'])
            xmax = float(box['xmax'])
            ymax = float(box['ymax'])
            # 计算 class x_center y_center width height
            width = xmax - xmin
            height = ymax - ymin
            # x_center y_center
            x_center = (xmax - width / 2) / max_width
            y_center = (ymax - height / 2) / max_height
            width_ratio = width / max_width
            height_ratio = height / max_height
            print(f"{filename}: {cls} {x_center} {y_center} {width_ratio} {height_ratio}")
            # 写文件
            get_file_stream(filename).write(f"{cls} {x_center} {y_center} {width_ratio} {height_ratio}\n")
            # break
        print(f"label dict: {json.dumps(label_dict)}")
    # print(content_json)


if __name__ == '__main__':
    try:
        main()
    finally:
        close_all_file_stream()