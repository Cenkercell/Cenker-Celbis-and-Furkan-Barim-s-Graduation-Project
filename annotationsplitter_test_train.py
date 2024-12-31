import os
import json
import random

def split_dataset(image_dir, annotations_path, train_ratio=0.8, output_dir="output"):
    """
    Verileri train ve test olarak ayırır.
    
    :param image_dir: Görsellerin bulunduğu klasör.
    :param annotations_path: JSON anotasyon dosyasının yolu.
    :param train_ratio: Eğitim verilerinin oranı (default: 0.8).
    :param output_dir: Çıktı dosyalarının kaydedileceği klasör.
    """
    # Klasör oluştur
    os.makedirs(output_dir, exist_ok=True)

    # Anotasyonları yükle
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # Tüm görüntü dosyalarını alın
    image_files = list(annotations.keys())
    random.shuffle(image_files)  # Görselleri karıştır
    
    # Verileri train ve test olarak ayır
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]

    # Eğitim ve test anotasyonlarını oluştur
    train_annotations = {key: annotations[key] for key in train_files}
    test_annotations = {key: annotations[key] for key in test_files}

    # Eğitim ve test JSON dosyalarını kaydet
    train_path = os.path.join(output_dir, "train_annotations.json")
    test_path = os.path.join(output_dir, "test_annotations.json")
    
    with open(train_path, "w") as f:
        json.dump(train_annotations, f, indent=4)
    with open(test_path, "w") as f:
        json.dump(test_annotations, f, indent=4)
    
    print(f"Eğitim verileri ({len(train_files)} görüntü) {train_path} dosyasına kaydedildi.")
    print(f"Test verileri ({len(test_files)} görüntü) {test_path} dosyasına kaydedildi.")

# Ana fonksiyon
if __name__ == "__main__":
    # Klasör ve dosya yolları
    image_dir = "dataset_jr/"  # Görsellerin bulunduğu klasör
    annotations_path = "output_annotations/annotations.json"  # Anotasyon dosyasının yolu
    output_dir = "output"  # Çıktı dosyalarının kaydedileceği klasör

    # Dataset'i ayır
    split_dataset(image_dir, annotations_path, train_ratio=0.8, output_dir=output_dir)
