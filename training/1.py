from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
    data=r"D:\Madhukar\YOLO\YAML\data_1.yaml",
    epochs=50,
    imgsz=320, rect=True,
    save_period=10,
    device='cuda'
)

if __name__ == "__main__":
    train()