from typing import Dict
import cv2
import numpy as np
import onnxruntime as ort

def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences

def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    print(np.array(keep).shape)
    return keep, keep_confidences

def filter_detections(results, thresh = 0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)

        return considerable_detections

class HVACDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = {
            0 : 'hvac_label',
            1 : 'hvac-rtu',
            2 : 'vent',
            3 : 'ven-unit'
        }
        
    def predict(self, image):
        self.input_img = self.preprocess(image)
        self.img_height, self.img_width = image.shape[:2]
        outputs = self.session.run(None, {self.input_name: self.input_img})        
        return self.postprocess(outputs)
    
    def preprocess(self, image):
        img = cv2.resize(image, (640, 640))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img
    
    def postprocess(self, outputs):
        results = outputs[0]
        results = results.transpose()
        results = filter_detections(results)
        # Si no hay detecciones, retorna lista vacía
        if results.size == 0 or len(results.shape) < 2:
            return []
        rescaled_results, confidences = rescale_back(results, self.img_width, self.img_height)
        boxes = []
        for res, conf in zip(rescaled_results, confidences):
            x1,y1,x2,y2, cls_id = res
            cls_id = int(cls_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = "{:.2f}".format(conf)
            box = [x1, y1, x2, y2, conf, self.class_names[cls_id]]
            boxes.append(box)
        return boxes






MODEL_PATHS = {
    "hvac": f"./models/yolo-hvac.onnx"
}

def load_models() -> Dict[str, HVACDetector]:
    """
    Load YOLO models from disk and return them in a dictionary.

    Returns:
        dict[str, YOLO]: Dictionary of models keyed by type ("n", "s", "m")
    """
    models = {}
    for model_type, path in MODEL_PATHS.items():
        models[model_type] = HVACDetector(path)
    return models
