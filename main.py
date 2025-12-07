import cv2
import numpy as np
import sys

# ================== AYARLAR ==================
video_source = 'mouse_video.mp4'
min_area = 500

# Beklenen gerçek fare sayısı
RAT_COUNT = 2           # 1, 2, 3... deneyine göre elle ayarla
MAX_RATS = RAT_COUNT    # Maksimum takip edilecek track sayısı

# Çizim Ayarları
line_intensity = 20      # İzin ne kadar hızlı belirginleşeceği
line_thickness = 2       # Çizgi kalınlığı

# Video boyutu (resize için)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Takip Ayarları
MAX_DIST = 80            # Bir framede fare en fazla kaç piksel hareket edebilir?

# Piksel -> cm dönüşümü (kalibrasyon biliyorsan burayı doldur)
# Örn: 1 px = 0.1 cm ise PIXEL_TO_CM = 0.1
PIXEL_TO_CM = None       # Kalibrasyon yoksa None bırak

# --- BUTON AYARLARI ---
btn_x, btn_y, btn_w, btn_h = 20, 20, 100, 40
stop_processing = False

# --- ARENA ROI (FİZİKSEL TABAN) ---
# mouse_video.mp4 için ayarlanmıştır.
ARENA_X_MIN = 190
ARENA_X_MAX = 450
ARENA_Y_MIN = 60
ARENA_Y_MAX = 400

# ratVideoNoldus.mp4 için ayarlar:
""" ARENA_X_MIN = 200
ARENA_X_MAX = 440
ARENA_Y_MIN = 80
ARENA_Y_MAX = 420
 """
# ================== RENK AYARLARI (HER TRACK İÇİN) ==================
# BGR formatında kutu ve nokta renkleri
TRACK_COLORS = [
    {"box": (255, 255, 0), "point": (0, 0, 255)},     # Track 0: Aqua / Mavi
    {"box": (0, 0, 255),   "point": (0, 255, 255)},   # Track 1: Kırmızı
    {"box": (0, 255, 255), "point": (255, 0, 0)},     # Track 2: Sarı
    {"box": (0, 165, 255), "point": (0, 128, 255)},   # Track 3: Turuncu
]


# ================== YARDIMCI FONKSİYONLAR ==================
def is_inside_arena(cx, cy):
    """
    Detections ONLY from the real floor / arena.
    STRICTLY within the defined ROI boundaries.
    No detection outside these boundaries.
    """
    return (
        ARENA_X_MIN <= cx <= ARENA_X_MAX and
        ARENA_Y_MIN <= cy <= ARENA_Y_MAX
    )


def mouse_handler(event, x, y, flags, param):
    """FINISH butonuna tıklayınca işlemi durdur."""
    global stop_processing
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x <= x <= btn_x + btn_w and btn_y <= y <= btn_y + btn_h:
            print("[BUTON] Bitir butonuna basıldı. Durduruluyor...")
            stop_processing = True


# ----------- METRİK FONKSİYONLARI -----------
def compute_total_distance(points, pixel_to_cm=None):
    """
    Bir farenin tüm trajectory'sinden toplam yol hesabı.
    points: [(x, y), ...]
    """
    if len(points) < 2:
        return {"pixels": 0.0, "cm": 0.0 if pixel_to_cm else None}

    dist_px = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dist_px += (dx ** 2 + dy ** 2) ** 0.5

    dist_cm = dist_px * pixel_to_cm if pixel_to_cm is not None else None
    return {"pixels": dist_px, "cm": dist_cm}


def compute_center_metrics(points, frame_width, frame_height, center_ratio=0.5):
    """
    Point'ların kaçı 'merkez bölge'de?
    center_ratio=0.5 -> genişlik ve yüksekliğin %50'si kadar merkez kare.
    """
    if not points:
        return 0.0, 0

    cx_min = frame_width * (1.0 - center_ratio) / 2.0
    cx_max = frame_width * (1.0 + center_ratio) / 2.0
    cy_min = frame_height * (1.0 - center_ratio) / 2.0
    cy_max = frame_height * (1.0 + center_ratio) / 2.0

    center_frames = 0
    for (x, y) in points:
        if cx_min <= x <= cx_max and cy_min <= y <= cy_max:
            center_frames += 1

    ratio = center_frames / len(points)
    return ratio, center_frames


def summarize_tracks(tracks, fps, frame_width, frame_height,
                     pixel_to_cm=None, center_ratio=0.5):
    """
    Her track için:
    - toplam frame
    - toplam süre (sn)
    - toplam mesafe (px / cm)
    - center-time ratio
    döndürür.
    """
    summaries = []

    for track_id, track in enumerate(tracks):
        points = track.get("points", [])
        if not points:
            continue

        total_frames = len(points)
        total_time_s = total_frames / fps if fps and fps > 0 else None

        dist_info = compute_total_distance(points, pixel_to_cm=pixel_to_cm)
        center_ratio_val, center_frames = compute_center_metrics(
            points, frame_width, frame_height, center_ratio=center_ratio
        )

        summaries.append({
            "track_id": track_id,
            "total_frames": total_frames,
            "total_time_s": total_time_s,
            "distance_pixels": dist_info["pixels"],
            "distance_cm": dist_info["cm"],
            "center_ratio": center_ratio_val,
            "center_frames": center_frames,
        })

    return summaries


# ================== BAŞLATMA ==================
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"[HATA] '{video_source}' bulunamadı.")
    sys.exit()

# FPS bilgisi (metrikler için)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0  # fallback

window_name = "Multi-Color Rat Tracker"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_handler)

# DÖRT AYRI TUVAL OLUŞTURUYORUZ (Her Fare için Bir Tuval)
canvas_aqua   = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)  # 1. Fare
canvas_red    = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)  # 2. Fare
canvas_yellow = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)  # 3. Fare
canvas_orange = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)  # 4. Fare

canvases = [
    canvas_aqua,
    canvas_red,
    canvas_yellow,
    canvas_orange,
]

# Her track için durum: aktif mi, son nokta ne, tüm noktalar listesi
tracks = [
    {"active": False, "last_point": None, "points": []}
    for _ in range(MAX_RATS)
]

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
final_result_frame = None

print("[BILGI] Analiz başladı... 1. Fare: MAVİ, 2. Fare: KIRMIZI, 3. Fare: SARI, 4. Fare: TURUNCU")
print("[BILGI] Arena ROI kullanılıyor, duvar yansımaları büyük oranda filtrelenecek.")

# ================== ANA DÖNGÜ ==================
while True:
    if stop_processing:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    roi = frame.copy()

    # Hareket Algılama
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Fareleri büyüklüğe göre sırala (En büyük kontur en başta)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # --- DETECTION LİSTESİ OLUŞTUR ---
    # Her detection: (cx, cy, x, y, w, h, area)
    # ONLY accept detections strictly INSIDE the arena
    detections = []
    for cnt in sorted_contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = int(x + w / 2), int(y + h / 2)

        # STRICT: Yalnızca arena içindeki detection'lar
        # Eğer arena dışında ise tamamen görmezden gel
        if not is_inside_arena(cx, cy):
            continue

        detections.append((cx, cy, x, y, w, h, area))

    # Çok fazla detection varsa, en büyük alanlı olanlardan başlayarak RAT_COUNT kadarını tut
    if len(detections) > RAT_COUNT:
        detections = sorted(detections, key=lambda d: d[6], reverse=True)[:RAT_COUNT]

    # --- TRACK ASSIGNMENT (ID KORUMA) ---
    track_assignment = {tid: None for tid in range(MAX_RATS)}
    used_detections = set()

    # 1) Aktif track'ler için en yakın detection'ı bul
    for tid, track in enumerate(tracks):
        if not track["active"] or track["last_point"] is None:
            continue

        tx, ty = track["last_point"]
        best_idx = None
        best_dist = None

        for det_idx, (cx, cy, x, y, w, h, area) in enumerate(detections):
            if det_idx in used_detections:
                continue

            dist = np.hypot(cx - tx, cy - ty)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = det_idx

        if best_idx is not None and best_dist is not None and best_dist <= MAX_DIST:
            track_assignment[tid] = best_idx
            used_detections.add(best_idx)

    # 2) Eşleşmemiş detection'lar için boş track aç
    for det_idx, det in enumerate(detections):
        if det_idx in used_detections:
            continue

        current_active = sum(1 for t in tracks if t["active"])
        if current_active >= RAT_COUNT:
            break

        # Boş bir track bul
        new_tid = None
        for tid, track in enumerate(tracks):
            if not track["active"]:
                new_tid = tid
                break

        if new_tid is None:
            break

        tracks[new_tid]["active"] = True
        track_assignment[new_tid] = det_idx
        used_detections.add(det_idx)

    # 3) Her track için çizim + iz güncelleme
    for tid, det_idx in track_assignment.items():
        if det_idx is None:
            continue

        cx, cy, x, y, w, h, area = detections[det_idx]
        current_point = (cx, cy)

        color_cfg = TRACK_COLORS[tid]
        box_color = color_cfg["box"]
        point_color = color_cfg["point"]

        # Kutu ve merkez çiz
        cv2.rectangle(roi, (x, y), (x + w, y + h), box_color, 2)
        cv2.circle(roi, current_point, 5, point_color, -1)

        # İz çiz
        last_pt = tracks[tid]["last_point"]
        if last_pt is not None:
            cv2.line(canvases[tid], last_pt, current_point, line_intensity, line_thickness)

        tracks[tid]["last_point"] = current_point
        tracks[tid]["points"].append(current_point)

    # ================== İZLERİ GÖRÜNTÜYE EKLE ==================
    # 1. AQUA İZLERİ EKLE
    mask_aqua_raw = np.clip(canvas_aqua, 0, 255).astype(np.uint8)
    mask_aqua_indices = mask_aqua_raw > 10
    if np.any(mask_aqua_indices):
        aqua_color = np.full_like(roi[mask_aqua_indices], [255, 255, 0], dtype=np.uint8)
        roi[mask_aqua_indices] = cv2.addWeighted(roi[mask_aqua_indices], 0.5, aqua_color, 0.5, 0)

    # 2. KIRMIZI İZLERİ EKLE
    mask_red_raw = np.clip(canvas_red, 0, 255).astype(np.uint8)
    mask_red_indices = mask_red_raw > 10
    if np.any(mask_red_indices):
        red_color = np.full_like(roi[mask_red_indices], [0, 0, 255], dtype=np.uint8)
        roi[mask_red_indices] = cv2.addWeighted(roi[mask_red_indices], 0.5, red_color, 0.5, 0)

    # 3. SARI İZLERİ EKLE
    mask_yellow_raw = np.clip(canvas_yellow, 0, 255).astype(np.uint8)
    mask_yellow_indices = mask_yellow_raw > 10
    if np.any(mask_yellow_indices):
        yellow_color = np.full_like(roi[mask_yellow_indices], [0, 255, 255], dtype=np.uint8)
        roi[mask_yellow_indices] = cv2.addWeighted(roi[mask_yellow_indices], 0.5, yellow_color, 0.5, 0)

    # 4. TURUNCU İZLERİ EKLE
    mask_orange_raw = np.clip(canvas_orange, 0, 255).astype(np.uint8)
    mask_orange_indices = mask_orange_raw > 10
    if np.any(mask_orange_indices):
        orange_color = np.full_like(roi[mask_orange_indices], [0, 165, 255], dtype=np.uint8)
        roi[mask_orange_indices] = cv2.addWeighted(roi[mask_orange_indices], 0.5, orange_color, 0.5, 0)

    # --- FINISH BUTONU ---
    cv2.rectangle(roi, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 0, 200), -1)
    cv2.putText(
        roi,
        "FINISH",
        (btn_x + 15, btn_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # (Opsiyonel) Arena ROI'yi görsel kontrol için çiz
    cv2.rectangle(
        roi,
        (ARENA_X_MIN, ARENA_Y_MIN),
        (ARENA_X_MAX, ARENA_Y_MAX),
        (0, 255, 0),
        1,
    )

    cv2.imshow(window_name, roi)
    final_result_frame = roi

    if cv2.waitKey(10) == ord('q'):
        break

# ================== KAYDETME ==================
print("[BILGI] İşlem bitti. Kaydediliyor...")
if final_result_frame is not None:
    cv2.imwrite("final_color_paths.png", final_result_frame)
    print("[BASARILI] Harita 'final_color_paths.png' olarak kaydedildi.")

    cv2.imwrite("final_frame.png", final_result_frame)
    print("[BASARILI] Son frame 'final_frame.png' olarak kaydedildi.")

    cv2.imshow("SONUC (Cikmak icin tusa bas)", final_result_frame)
    cv2.waitKey(0)

# ================== METRİKLERİ HESAPLA ==================
print("\n[METRIKLER] Fare bazında özet:")

summaries = summarize_tracks(
    tracks,
    fps=fps,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    pixel_to_cm=PIXEL_TO_CM,
    center_ratio=0.5,  # Arena'nın %50'si merkez kabul ediliyor
)

if not summaries:
    print("Takip edilen fare bulunamadı veya trajectory boş.")
else:
    for s in summaries:
        tid = s["track_id"]
        print(f"\n--- Fare {tid + 1} ---")
        print(f"Toplam frame: {s['total_frames']}")
        if s["total_time_s"] is not None:
            print(f"Toplam süre: {s['total_time_s']:.2f} sn (fps={fps:.2f})")
        print(f"Toplam mesafe (px): {s['distance_pixels']:.2f}")
        if s["distance_cm"] is not None:
            print(f"Toplam mesafe (cm): {s['distance_cm']:.2f}")
        print(f"Merkezde kalma ratio: {s['center_ratio']:.3f} "
              f"({s['center_frames']} frame merkezde)")

cap.release()
cv2.destroyAllWindows()
