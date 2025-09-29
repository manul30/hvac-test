import numpy as np
import cv2
from .model_loader import HVACDetector

def process_image(file_bytes: bytes, model: HVACDetector) -> bytes:
    """
    Run YOLO on an image and return the annotated result as JPEG bytes.

    Args:
        file_bytes: Raw image data.
        model: YOLO model object.

    Returns:
        JPEG-encoded annotated image as bytes.
    """
    img_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    results = model.predict(image)

    # Suponiendo que results contiene las cajas y clases detectadas
    annotated = image.copy()
    for box in results:  # Ajusta según el formato de salida de postprocess
        x1, y1, x2, y2, conf, cls = box  # ejemplo: [x1, y1, x2, y2, conf, class]
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(annotated, f"{int(cls)} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    _, encoded = cv2.imencode(".jpg", annotated)
    return encoded.tobytes()



#  Bytes Image -> Numpy Array -> OpenCV Image -> YOLO Model -> Annotated Image -> JPEG Bytes -> Return Bytes