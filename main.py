import cv2
import numpy as np
import sys

# --- AYARLAR ---
video_source = 'mouse_video.mp4'
min_area = 500

# Çizgi Ayarları
line_thickness = 2      # Çizgi kalınlığı
line_intensity = 15     # Çizgi koyulaşma hızı (5-20 arası)

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"[HATA] '{video_source}' bulunamadı.")
    sys.exit()

ret, first_frame = cap.read()
if not ret: sys.exit()
first_frame = cv2.resize(first_frame, (640, 480))

# Yörünge tuvali (float32 tipinde toplama işlemi için)
trajectory_canvas = np.zeros((480, 640), dtype=np.float32)

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

last_point = None
final_result_frame = None

print("[BILGI] Analiz başladı... Sık geçilen yollar PARLAK görünecek.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))
    roi = frame.copy()

    # Maskeleme
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area > max_area:
            max_area = area
            largest_contour = cnt
    
    current_point = None

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        current_point = (cx, cy)

        # Anlık takip görseli (Yeşil kutu, Kırmızı nokta)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(roi, current_point, 5, (0, 0, 255), -1)

        # Çizgi ekleme mantığı
        if last_point is not None:
            cv2.line(trajectory_canvas, last_point, current_point, (line_intensity), line_thickness)

    if current_point is not None:
        last_point = current_point

    # --- DÜZELTİLMİŞ GÖRSELLEŞTİRME KISMI ---
    
    # 1. Tuvali normalize et (0-255 arası)
    heatmap_norm = np.clip(trajectory_canvas, 0, 255).astype(np.uint8)
    
    # 2. Renklendirme yap
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_HOT)
    
    # 3. Maskeyi oluştur (Siyah olmayan, yani çizgi olan yerler)
    mask_ind = heatmap_norm > 0
    
    # 4. HATA VERMEYEN GÜVENLİ BİRLEŞTİRME:
    # Eğer ekranda çizgi varsa işlem yap
    if np.any(mask_ind):
        # Önce tüm ekranı %40 Orijinal + %60 Renkli Harita olarak karıştır
        blended_frame = cv2.addWeighted(roi, 0.4, heatmap_color, 0.6, 0)
        
        # Sonra sadece çizgi olan kısımları ana görüntüye kopyala
        # (Bu yöntem, önceki slicing hatasını engeller)
        roi[mask_ind] = blended_frame[mask_ind]

    cv2.imshow("Agirlikli Yorunge Takibi", roi)
    final_result_frame = roi

    if cv2.waitKey(10) == ord('q'): break

# --- KAYDETME ---
print("[BILGI] Video bitti. Sonuç kaydediliyor...")

if final_result_frame is not None:
    cv2.imwrite("path_density_map.png", final_result_frame)
    print("[BAŞARILI] Harita 'path_density_map.png' olarak kaydedildi.")
    
    cv2.imshow("SONUC (Cikmak icin tusa bas)", final_result_frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()