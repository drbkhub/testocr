
import cv2
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

from utils import map_cls_to_boxes

# Load the pre-trained model
filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)



def layout(image_path):
    # Perform prediction
    det_res = model.predict(
        image_path,   # Image to predict
        imgsz=1024,        # Prediction image size
        conf=0.2,          # Confidence threshold
        device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    annotated_frame = det_res[0].plot(pil=True, line_width=3, font_size=20)
    cv2.imwrite("result.png", annotated_frame)
    return map_cls_to_boxes(det_res[0].boxes)
