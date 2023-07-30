import cv2
import numpy as np
from ultralytics import YOLO

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        for seg in result.masks.segments:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        # Load COCO class labels
        coco_labels = load_coco_labels("D:\\yolo_segmentation\\coco_labels.txt")

        # Combine the results into a list of dictionaries
        detections = []
        for bbox, class_id, seg, score in zip(bboxes, class_ids, segmentation_contours_idx, scores):
            (x, y, x2, y2) = bbox
            class_name = coco_labels[int(class_id)]
            
            detections.append({
                'class_name': class_name,
                'class_id': int(class_id),
                'confidence': float(score),
                'bbox': [int(x), int(y), int(x2), int(y2)],
                'segmentation': seg.tolist(),
            })

        return detections

def load_coco_labels(filename):
    with open(filename, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

if __name__ == "__main__":
    model_path = "yolov8m-seg.pt"
    ys = YOLOSegmentation(model_path)

    image_path = "D:\\yolo_segmentation\\images\\dog.jpg"
    img = cv2.imread(image_path)

    # Check if the image has been loaded successfully
    if img is None:
        raise ValueError("Image not found or cannot be loaded.")

    # Resize the image
    img = cv2.resize(img, None, fx=0.7, fy=0.7)

    # Perform detection and segmentation on the image
    detections = ys.detect(img)

    # Now you can process the detection and segmentation results
    for detection in detections:
        bbox = detection['bbox']
        class_id = detection['class_id']
        seg = detection['segmentation']
        score = detection['confidence']

        (x, y, x2, y2) = bbox
        cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

        # Reshape the seg variable to the expected shape (N, 1, 2) for cv2.polylines
        seg = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [seg], True, (255, 0, 0), 2)

        cv2.putText(img, f"{class_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    # Save the image with detections and segmentations
    output_image_path = "D:\\yolo_segmentation\\outputs\\output_image.jpg"
    cv2.imwrite(output_image_path, img)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
