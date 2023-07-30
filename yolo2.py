import cv2
import os
import json
import numpy as np
from ultralytics import YOLO

def load_coco_labels(filename):
    with open(filename, 'r') as f:
        labels = f.read().strip().split('\n')
    return labels

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

def save_results_to_json(results, output_dir, image_name):
    output_path = os.path.join(output_dir, f'detections_{image_name}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    model_path = "yolov8m-seg.pt"
    ys = YOLOSegmentation(model_path)

    image_paths = [
        "D:\\Yolo_NAS\\yolo_nas\\images\\mfc1.jpeg",
        "D:\\Yolo_NAS\\yolo_nas\\images\\mfc2.jpeg",
        "D:\\Yolo_NAS\\yolo_nas\\images\\mfc33.jpeg"
    ]

    output_dir = "D:\\yolo_segmentation\\output"

    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.9, fy=0.9)  # Use fx=0.9 and fy=0.9 instead of 0.9 and 0.
        ys = YOLOSegmentation("yolov8m-seg.pt")
        detections = ys.detect(img)

        # Save the detections in JSON format for each image
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        save_results_to_json(detections, output_dir, image_name)

        # Visualize the detections on the image
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            seg = detection['segmentation']
            score = detection['confidence']

            if score > 0.5:  # Filter detections based on confidence
                (x, y, x2, y2) = bbox
                class_name = detection['class_name']

                cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)

                # Reshape the seg variable to the expected shape (N, 1, 2) for cv2.polylines
                seg = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [seg], True, (255, 0, 0), 2)

                cv2.putText(img, f"{class_name} ({score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Save the image with visualizations
        output_image_path = os.path.join(output_dir, f'predicted_image_{image_name}.jpg')
        cv2.imwrite(output_image_path, img)

        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("Processing complete.")



#the code was working it put bounding boxs and polylines the predicted images are saved in the current directory and the json files are also but the image is not segmented
