from PIL import Image, ImageDraw


def map_cls_to_boxes(results_obj, names=False) -> list[dict[str | int, tuple[float, float, float, float]]]:
    names = {
        0: 'title', 
        1: 'plain text', 
        2: 'abandon', 
        3: 'figure', 
        4: 'figure_caption', 
        5: 'table', 
        6: 'table_caption', 
        7: 'table_footnote', 
        8: 'isolate_formula', 
        9: 'formula_caption',
        }
    cls_tensor = results_obj.cls
    xyxy_tensor = results_obj.xyxy
    

    result = []
    for cls_id, box in zip(cls_tensor, xyxy_tensor):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = list(map(lambda x: round(float(x), 1), box, ))
        if names:
            result.append({"type": names[cls_id], "bbox": (x1, y1, x2, y2)})
        else:
            result.append({"type": cls_id, "bbox": (x1, y1, x2, y2)})
    
    return result


def is_intersects(bbox0, bbox1):
    x_overlap = not (bbox0[2] < bbox1[0] or bbox1[2] < bbox0[0])
    y_overlap = not (bbox0[3] < bbox1[1] or bbox1[3] < bbox0[1])
    return x_overlap and y_overlap


def visualize(data) -> Image.Image:
    pil_image = Image.open("result.png")
    draw = ImageDraw.Draw(pil_image)
    for item in data:
        for text in item.get("texts", []):
            draw.rectangle(text["bbox"], outline="blue", width=3)
    pil_image.save("result.png")