
import argparse
import json
import easyocr

from image import extract_images_from_pdf
from layout import layout
from ocr import ocr
from utils import is_intersects, visualize





# # Load the pre-trained model
# filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
# model = YOLOv10(filepath)





# extract_images_from_pdf(file, image_path, page_index)


# # Perform prediction
# det_res = model.predict(
#     image_path,   # Image to predict
#     imgsz=1024,        # Prediction image size
#     conf=0.2,          # Confidence threshold
#     device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
# )

# # print(dir(det_res[0].boxes))

# print(map_cls_to_boxes(det_res[0].boxes))


# print(det_res)
# print(type(det_res[0]))
# Annotate and save the result
# annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
# annotated_frame:doclayout_yolo.engine.results.Results   = det_res[0].
# cv2.imwrite("result.jpg", annotated_frame)



class OCR:
    def __init__(self, filepath, langs = None, page_index = None):
        self.filepath = filepath
        self.langs = langs if langs else ["en"]
        self.page_index = page_index

        if self.page_index is None:
            print(f"Распознавание изображения '{self.filepath}'")
        else:
            print(f"Распознавание страницы {self.page_index} из файла '{self.filepath}'")
        print("Используемые языки:", ", ".join(self.langs))
        try:
            self.reader = easyocr.Reader(self.langs)
        except ValueError as e:
            raise ValueError(f"Не поддерживаемый язык '{str(e)}'")

    @property
    def _image_path(self):
        # изображение
        if self.page_index is None:
            return self.filepath
        
        else:
            temp_path = "temp_page.png"
            extract_images_from_pdf(self.filepath, temp_path, self.page_index)
            return temp_path

    def _ocr(self):
        return ocr(self._image_path, self.reader)
    
    def _layout(self):
        return layout(self._image_path)

    def process(self):
        ocr_result = self._ocr()
        layout_result = self._layout()

        for item in layout_result:
            l_bbox = item["bbox"]
            for i in range(len(ocr_result)-1, -1, -1):
                ocr_item = ocr_result[i]
                o_bbox = ocr_item["bbox"]
                # определяем тип текста
                if is_intersects(l_bbox, o_bbox):
                    item["texts"] = item.get("texts", [])
                    item["texts"].append(ocr_item)
                    # удаляем из последующей проверки
                    ocr_result.remove(ocr_item)

        return layout_result
        
    def json(self):
        return json.dumps(self.process(), ensure_ascii=False, indent=2)


# file = "/home/boss/pdf/test2.pdf"
# page_index = 1
# langs = ["en"]

# file_ocr = OCR(file, langs, page_index)
# # print(file_ocr.ocr())
# print(file_ocr.process())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Распознавание макета страницы и распознавание текста с помощью OCR изображения или страницы pdf (нужно указать индекс страницы)")
    
    parser.add_argument("--file", type=str, required=True, help="Путь к pdf или к изображению")
    parser.add_argument("--index", type=int, default=None, help="Индекс страницы. Только для PDF")
    parser.add_argument("--langs", type=str, nargs="+", default=["en"], help="Языки страницы (EasyOCR). Можно указать несколько.")
    args = parser.parse_args()

    file_ocr = OCR(args.file, args.langs, args.index)
    data = file_ocr.process()
    data.sort(key=lambda x: x["bbox"][1])
    result_json = json.dumps(data, ensure_ascii=False, indent=2)
    with open("result.json", "w") as f:
        f.write(result_json)
    print(result_json)
    visualize(data)
