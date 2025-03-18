from sahi.predict import predict

predict(
    model_type="yolov8",
    model_path="/home/ps/DiskA/shijiale/v8-improve/ultralytics_improve-mask/runs/detect/train10-k=11-300/weights/best.pt",
    model_device="cuda:2",  # or 'cpu'
    model_confidence_threshold=0.1,
    source="/home/ps/DiskA/shijiale/rescue-YOLOV8/runs-AIR/detect/train18/weights/failed-pic/test_BLI_0002.jpg",
    slice_height=1000,
    slice_width=1000,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    export_crop=False,
)