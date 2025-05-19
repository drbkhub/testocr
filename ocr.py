
def ocr(image_path, reader):
    # reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory
    result = reader.readtext(image_path, width_ths=0.4)
    result_items = []
    for item in result:
        coords = item[0]
        x0 = float(round(min(c[0] for c in coords), 1))
        y0 = float(round(min(c[1] for c in coords), 1))
        x1 = float(round(max(c[0] for c in coords), 1))
        y1 = float(round(max(c[1] for c in coords), 1))
        result_items.append(
            {
                "bbox": (x0, y0, x1, y1),
                "text": item[1],
            }
        )
    return result_items
