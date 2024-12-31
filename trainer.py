import os
import torch
import torchvision

from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torch.utils.data import DataLoader, Dataset
import json
import cv2
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms.functional as F

# Veri seti için özel bir Dataset sınıfı
class ResistorDataset(Dataset):
    def __init__(self, data_dir, annotations_path):
        self.data_dir = data_dir
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation = self.annotations[img_name]
        boxes = torch.tensor(annotation['boxes'], dtype=torch.float32)
        labels = torch.tensor(annotation['labels'], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        image = F.to_tensor(image)
        return image, target

# Model oluşturma fonksiyonu
def create_model(num_classes=2, size=300):
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    in_channels = torchvision.models.detection._utils.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

# Eğitim döngüsü
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0
    for images, targets in tqdm(data_loader, desc=f"Epoch {epoch}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    return epoch_loss / len(data_loader)

# Ana fonksiyon
def main():
    data_dir = "dataset_jr/"
    annotations_path = "output/train_annotations.json"  # Eğitim verileri
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    batch_size = 4
    num_epochs = 40
    lr = 0.0001
    dataset = ResistorDataset(data_dir, annotations_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    model = create_model(num_classes=num_classes, size=300)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, data_loader, device, epoch + 1)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), "temp_canan.pth")
    print("Model kaydedildi.")

if __name__ == "__main__":
    main()
