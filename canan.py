import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

def build_model(weights_path, num_classes=2, size=300):
    """
    SSD300 VGG16 modelini yükler, yapılandırır ve önceden eğitilmiş ağırlıkları uygular.
    
    Args:
        weights_path (str): Model ağırlıklarının yolu.
        num_classes (int): Sınıf sayısı.
        size (int): Görüntü boyutu (ör: 300x300).
        
    Returns:
        torch.nn.Module: Yüklenmiş ve yapılandırılmış model.
    """
    # COCO veri kümesi ağırlıkları ile SSD300 VGG16 modeli
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    in_channels = torchvision.models.detection._utils.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    
    # Sınıflandırma başlığını yeniden yapılandır
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes
    )
    
    # Eğitilmiş ağırlıkları yükle
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Değerlendirme modu
    return model
