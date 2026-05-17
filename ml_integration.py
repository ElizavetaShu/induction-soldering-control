from ultralytics import YOLO
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Цвета для разных классов объектов (для красивого отображения)
SEG_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
]

def import_model(path: str="yolo11n.pt"):
        model = YOLO(path)
        return model

def import_seg_model(path: str="yolo11n-seg.pt"):
    """Загружает модель сегментации"""
    model = YOLO(path)
    return model
    
def inference_model(model, classNames:dict, frame:cv2.typing.MatLike, device: str = 'cpu')->cv2.typing.MatLike:
    start = time.time()
    prediction = model.predict(frame, device=device, verbose=False, imgsz=320)
    
    for r in prediction:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # class confidence
            confidence = math.ceil((box.conf[0]*100))/100
            
            # class name
            cls = int(box.cls[0])
            color = (255/len(classNames)*cls, 255/len(classNames)*(len(classNames)-cls), 255*len(classNames)/(cls+1))
            
            # object details
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, classNames[cls], [x1, y1-10], font, fontScale, color, thickness)
            cv2.putText(frame, str(confidence), [x2-80, y1+30], font, fontScale, color, thickness)

    elapsed = round((time.time() - start) * 1000, 1)  # мс
    return frame, elapsed


def inference_segmentation(model, classNames: dict, frame: np.ndarray, device='cpu') -> tuple:
    start = time.time()
    prediction = model.predict(frame, verbose=False, device=device, imgsz=480)

    for r in prediction:
        if r.masks is None or len(r.masks.xy) == 0:
            continue

        # r.masks.xy — список контуров: [(N,2), (M,2), ...]
        for i, contour in enumerate(r.masks.xy):
            contour = contour.astype(np.int32)
            if len(contour) == 0:
                continue

            # Класс и уверенность
            if i < len(r.boxes.cls):
                cls = int(r.boxes.cls[i])
                conf = r.boxes.conf[i]
                color = SEG_COLORS[cls % len(SEG_COLORS)]
                label = f"{classNames[cls]} {conf:.2f}"
            else:
                color = (255, 255, 255)
                label = "object"

            # Полупрозрачная маска
            overlay = frame.copy()
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)

            # Контурная линия
            cv2.polylines(frame, [contour], True, color, 2)

            # Подпись (✅ ИСПРАВЛЕНО ЗДЕСЬ)
            x, y = contour[0]  # contour[0] уже массив [x, y]
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    elapsed = round((time.time() - start) * 1000, 1)
    return frame, elapsed


# <<< НОВОЕ: простая отрисовка времени на кадре (добавить перед draw_gird) >>>
def draw_timing_info(frame: cv2.typing.MatLike, det_sec: float, seg_sec: float) -> cv2.typing.MatLike:
    """Рисует время выполнения моделей в углу кадра (в секундах)"""
    h, w = frame.shape[:2]

    # Параметры отображения
    x_offset = 15
    y_offset = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Фон под текстом (полупрозрачный)
    overlay = frame.copy()
    box_height = 70
    box_width = 200
    cv2.rectangle(overlay,
                  (x_offset - 10, y_offset - 20),
                  (x_offset + box_width, y_offset + box_height - 10),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    # Заголовок
    cv2.putText(frame, "⏱️  Время обработки:",
                (x_offset, y_offset),
                font, font_scale, (255, 255, 255), thickness)

    # Детекция (зелёный)
    det_color = (0, 255, 0) if det_sec < 0.05 else (0, 200, 200) if det_sec < 0.1 else (0, 0, 255)
    cv2.putText(frame, f"Detect: {det_sec:.3f} с",
                (x_offset, y_offset + 25),
                font, font_scale, det_color, thickness)

    # Сегментация (фиолетовый)
    seg_color = (255, 100, 255) if seg_sec < 0.1 else (100, 100, 255)
    cv2.putText(frame, f"Segment: {seg_sec:.3f} с",
                (x_offset, y_offset + 50),
                font, font_scale, seg_color, thickness)

    # Итого (если обе модели включены)
    if det_sec > 0 and seg_sec > 0:
        total = det_sec + seg_sec
        total_color = (0, 255, 255) if total < 0.15 else (0, 100, 255)
        cv2.putText(frame, f"Total: {total:.3f} с",
                    (x_offset, y_offset + 75),
                    font, font_scale, total_color, thickness)

    return frame

def draw_gird(frame:cv2.typing.MatLike)->cv2.typing.MatLike:
    h, w = frame.shape[:2]
    mod_1 = 1
    mod_2 = 5
    cv2.line(frame, (0, int(h*mod_1/mod_2)), (w, int(h*mod_1/mod_2)), (255, 255, 255), thickness=1)
    cv2.line(frame, (0, int(h*(mod_2 - mod_1)/mod_2)), (w, int(h*(mod_2 - mod_1)/mod_2)), (255, 255, 255), thickness=1)
    cv2.line(frame, (int(w*mod_1/mod_2), 0), (int(w*mod_1/mod_2), h), (255, 255, 255), thickness=1)
    cv2.line(frame, (int(w*(mod_2 - mod_1)/mod_2), 0), (int(w*(mod_2 - mod_1)/mod_2), h), (255, 255, 255), thickness=1)
    
    return frame


if __name__ == "__main__":
    model = import_model('yolo11n.pt')
    frame = cv2.imread('D:/Python/jupiter/bus.jpg')
    results = inference_model(model, model.names, frame)
    cv2.imshow("bus", results)
    cv2.waitKey()