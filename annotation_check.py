import os
import cv2
import json
from tqdm import tqdm

# Klasör yolları
uncropped_dir = "dataset_jr/"
annotation_path = "output_annotations/annotations.json"

def draw_annotations(image, boxes):
    """
    Görüntü üzerinde kutucukları çizer.
    """
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Yeşil kutu
    return image

def visualize_annotations(uncropped_dir, annotation_path):
    """
    Anotasyonları kullanarak görüntüleri kareye alıp sırayla gösterir.
    """
    # JSON dosyasını yükle
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    
    uncropped_files = [f for f in os.listdir(uncropped_dir) if f.endswith(('.jpg', '.png'))]
    
    for file_name in tqdm(uncropped_files, desc="Visualizing Annotations"):
        if file_name not in annotations:
            print(f"Annotation not found for {file_name}, skipping...")
            continue
        
        # Görüntü yolunu al
        uncropped_path = os.path.join(uncropped_dir, file_name)
        image = cv2.imread(uncropped_path)
        
        # Anotasyonları al
        boxes = annotations[file_name]["boxes"]
        
        # Anotasyonları çiz
        annotated_image = draw_annotations(image, boxes)
        
        # Görüntüyü göster
        cv2.imshow("Annotated Image", annotated_image)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Ana fonksiyon
if __name__ == "__main__":
    visualize_annotations(uncropped_dir, annotation_path)
