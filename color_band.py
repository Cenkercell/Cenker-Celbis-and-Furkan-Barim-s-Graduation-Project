import cv2
import numpy as np
import json
import torch
from canan import build_model

# Renk kodu tablosu
COLOR_CODE_TABLE = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "gray": 8,
    "white": 9
}

def load_hsv_values(file_name="hsv_values.json"):
    with open(file_name, "r") as file:
        return json.load(file)

def calculate_resistance(colors):
    """
    Verilen üç renk koduna göre direnç değeri hesapla.

    Args:
        colors (list): Renk isimleri listesi (örneğin: ["red", "yellow", "brown"]).

    Returns:
        int: Direnç değeri (ohm).
    """
    if len(colors) < 3:
        print("Yeterli renk algılanamadı!")
        return None

    try:
        first_digit = COLOR_CODE_TABLE[colors[0]]
        second_digit = COLOR_CODE_TABLE[colors[1]]
        multiplier = 10 ** COLOR_CODE_TABLE[colors[2]]
        return (first_digit * 10 + second_digit) * multiplier
    except KeyError:
        print("Bilinmeyen renk algılandı!")
        return None

def draw_bands_with_boxes(frame, bands, band_positions):
    for color, (x1, y1, x2, y2) in zip(bands, band_positions):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_and_draw_resistor_bands(frame, model, device, hsv_values):
    image_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = np.transpose(image_tensor, (2, 0, 1)) / 255.0
    image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    if len(predictions['scores']) > 0:
        max_score_idx = torch.argmax(predictions['scores']).item()
        x_min, y_min, x_max, y_max = map(int, predictions['boxes'][max_score_idx].tolist())

        cropped_frame = frame[y_min:y_max, x_min:x_max]

        band_positions = []
        band_color_mapping = []

        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
        height, width = cropped_frame.shape[:2]

        for color_name, bounds in hsv_values.items():
            lower = np.array(bounds["lower"], dtype="uint8")
            upper = np.array(bounds["upper"], dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Kutu içinde kutu algılamayı önlemek için
            valid_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = cv2.contourArea(contour)

                # Diğer konturların kapsama durumunu kontrol et
                is_nested = any(cv2.boundingRect(other)[0] < x and cv2.boundingRect(other)[1] < y and
                                cv2.boundingRect(other)[0] + cv2.boundingRect(other)[2] > x + w and
                                cv2.boundingRect(other)[1] + cv2.boundingRect(other)[3] > y + h
                                for other in contours if contour is not other)

                if not is_nested and 0.2 < aspect_ratio < 0.6 and height * 0.2 < h < height * 0.9 and area > 50:
                    valid_contours.append(contour)

            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                band_positions.append((x_min + x, y_min + y, x_min + x + w, y_min + y + h))
                band_color_mapping.append(color_name)

        if band_positions and band_color_mapping:
            band_positions, band_color_mapping = zip(
                *sorted(zip(band_positions, band_color_mapping), key=lambda pair: pair[0][0])
            )

            draw_bands_with_boxes(frame, band_color_mapping, band_positions)

            print(f"Algılanan Renkler: {', '.join(band_color_mapping[:3])}")

            resistance_value = calculate_resistance(band_color_mapping[:3])
            if resistance_value is not None:
                print(f"Direnç Değeri: {resistance_value} ohm")
        else:
            print("Bantlar algılanamadı veya eşleşen boyutlar bulunamadı.")

def main():
    weights_path = "canan.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hsv_values = load_hsv_values()

    model = build_model(weights_path, num_classes=2)
    model.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Tahmin işlemi başladı. Çıkmak için 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı!")
            break

        detect_and_draw_resistor_bands(frame, model, device, hsv_values)

        cv2.imshow("Direnç Bantları", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Tahmin işlemi sonlandırılıyor.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
