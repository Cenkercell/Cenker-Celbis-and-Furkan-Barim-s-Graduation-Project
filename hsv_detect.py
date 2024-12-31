import cv2
import numpy as np
import json
import torch
from canan import build_model


def nothing(x):
    pass


def load_hsv_values(file_name="hsv_values.json"):
    """
    JSON dosyasından HSV aralıklarını yükle.
    Mevcut dosya yoksa boş bir sözlük döndür.
    """
    try:
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"{file_name} dosyası bulunamadı, yeni bir dosya oluşturulacak.")
        return {}
    except json.JSONDecodeError:
        print(f"{file_name} dosyası okunamadı, JSON formatı hatalı. Yeni bir dosya oluşturulacak.")
        return {}


def save_hsv_values(values, file_name="hsv_values.json"):
    """
    HSV değerlerini JSON dosyasına kaydeder.
    """
    try:
        with open(file_name, "w") as file:
            json.dump(values, file, indent=4)
        print(f"HSV değerleri {file_name} dosyasına başarıyla kaydedildi!")
    except Exception as e:
        print(f"HSV değerleri kaydedilirken hata oluştu: {e}")


def main():
    # Canan modelini yükle
    weights_path = "canan.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(weights_path, num_classes=2)
    model.to(device)

    # Mevcut HSV değerlerini yükle
    hsv_values = load_hsv_values()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    # Trackbar'lar ile HSV belirleme
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H_min", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("H_max", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("S_min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("S_max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V_min", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("V_max", "Trackbars", 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı!")
            break

        # Canan modeli ile bölge tespiti
        image_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = np.transpose(image_tensor, (2, 0, 1)) / 255.0
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]

        if len(predictions['scores']) > 0:
            max_score_idx = torch.argmax(predictions['scores']).item()
            x_min, y_min, x_max, y_max = map(int, predictions['boxes'][max_score_idx].tolist())

            # Tespit edilen bölgeyi kırp
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # HSV uzayına dönüştür
            hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

            # Trackbar değerlerini oku
            h_min = cv2.getTrackbarPos("H_min", "Trackbars")
            h_max = cv2.getTrackbarPos("H_max", "Trackbars")
            s_min = cv2.getTrackbarPos("S_min", "Trackbars")
            s_max = cv2.getTrackbarPos("S_max", "Trackbars")
            v_min = cv2.getTrackbarPos("V_min", "Trackbars")
            v_max = cv2.getTrackbarPos("V_max", "Trackbars")

            # Maske oluştur
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv, lower, upper)

            # Maskeyi uygula ve sonuçları göster
            result = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
            cv2.imshow("Cropped Frame", cropped_frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)

            # 's' tuşuna basıldığında aralıkları kaydet
            if cv2.waitKey(1) & 0xFF == ord('s'):
                color_name = input("Renk adı girin: ").strip()
                if color_name:
                    hsv_values[color_name] = {
                        "lower": [int(h_min), int(s_min), int(v_min)],
                        "upper": [int(h_max), int(s_max), int(v_max)]
                    }
                    save_hsv_values(hsv_values)
                else:
                    print("Renk adı boş olamaz.")
        else:
            print("Canan modeli hiçbir bölge algılayamadı!")
            cv2.imshow("Frame", frame)

        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
