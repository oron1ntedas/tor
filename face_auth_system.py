import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
from face_functionality import HandController

script_dir = os.path.dirname(os.path.abspath(__file__))

# Параметры YOLO
CONFIG_PATH = os.path.join(script_dir, "models/yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(script_dir, "models/yolov4-tiny.weights")
CONFIDENCE_THRESHOLD = 0.701
NMS_THRESHOLD = 0.4

# Загрузка YOLO
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Инициализация MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)

def load_face_data(name):
    data_dir = os.path.join(script_dir, "face_data")
    # Ищем все файлы с именем пользователя
    face_files = [f for f in os.listdir(data_dir) if f.startswith(f"{name}_") and f.endswith('.pkl')]
    
    if not face_files:
        print(f"❌ Не найдены образцы лица для пользователя {name}")
        return None
    
    saved_landmarks = []
    for file_name in face_files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            landmarks = pickle.load(f)
            saved_landmarks.append(landmarks)
            print(f"✅ Загружен образец {file_name}")
    
    return saved_landmarks

def normalize_landmarks(landmarks):
    center = np.mean(landmarks, axis=0)
    normalized = landmarks - center
    scale = np.max(np.abs(normalized))
    if scale > 0:
        normalized = normalized / scale
    return normalized

def compare_faces(saved_landmarks_list, current_landmarks, threshold=0.15):
    if current_landmarks is None:
        print("❌ Текущие точки лица отсутствуют")
        return False
    
    best_match = False
    min_distance = float('inf')
    
    for i, saved_landmarks in enumerate(saved_landmarks_list):
        if saved_landmarks is None:
            continue
            
        landmarks1_norm = normalize_landmarks(saved_landmarks)
        landmarks2_norm = normalize_landmarks(current_landmarks)
        
        distances = np.sqrt(np.sum((landmarks1_norm - landmarks2_norm) ** 2, axis=1))
        median_distance = np.median(distances)
        
        if median_distance < min_distance:
            min_distance = median_distance
            best_match = median_distance < threshold
    
    # Выводим отладочную информацию
    print(f"Минимальное расстояние между точками: {min_distance:.3f}")
    print(f"Порог сравнения: {threshold}")
    print(f"Результат сравнения: {'Совпадает' if best_match else 'Не совпадает'}")
    
    return best_match

def detect_faces(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width * 0.8)
                h = int(detection[3] * height * 0.8)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

def get_face_landmarks(frame, face_box):
    x, y, w, h = face_box
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    results = face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    return None

def main():
    name = input("Введите имя для проверки лица: ")
    saved_landmarks_list = load_face_data(name)
    if saved_landmarks_list is None:
        return

    print(f"✅ Загружено {len(saved_landmarks_list)} образцов лица {name}")
    print("Программа запущена. Нажмите ESC для выхода.")
    print("Функционал будет доступен только при распознавании сохраненного лица.")

    hand_controller = HandController()
    cap = cv2.VideoCapture(0)
    
    # Устанавливаем разрешение камеры 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру.")

    last_face_count = 0
    last_recognized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = frame.copy()
        
        boxes = detect_faces(processed_frame)
        face_recognized = False
        current_face_count = len(boxes)

        if current_face_count != last_face_count:
            hand_controller.deactivate()
            print(f"Количество лиц изменилось с {last_face_count} на {current_face_count}. Функционал отключен.")

        overlay = processed_frame.copy()
        
        for box in boxes:
            current_landmarks = get_face_landmarks(processed_frame, box)
            if current_landmarks is not None:
                print("\n--- Проверка лица ---")
                if compare_faces(saved_landmarks_list, current_landmarks):
                    face_recognized = True
                    for landmark in current_landmarks:
                        px = int(landmark[0] * box[2]) + box[0]
                        py = int(landmark[1] * box[3]) + box[1]
                        cv2.circle(overlay, (px, py), 1, (255, 255, 255), -1)
                else:
                    print("Лицо не соответствует ни одному из сохраненных образцов")

        # Накладываем полупрозрачные точки
        cv2.addWeighted(overlay, 0.5, processed_frame, 0.5, 0, processed_frame)

        # Отображаем статус
        status_color = (0, 255, 0) if face_recognized else (0, 0, 255)
        cv2.putText(processed_frame, 
                   f"Статус: {'Распознано' if face_recognized else 'Не распознано'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(processed_frame, 
                   f"Лиц в кадре: {current_face_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Управление функционалом
        if face_recognized and current_face_count == 1:
            if not hand_controller.is_functionality_active():
                print("✅ Запуск функционала управления...")
                hand_controller.activate()
            processed_frame = hand_controller.process_frame(processed_frame)
        else:
            if hand_controller.is_functionality_active():
                hand_controller.deactivate()
                print("❌ Функционал отключен")

        cv2.imshow("Face Recognition & Hand Control", processed_frame)
        
        last_face_count = current_face_count
        last_recognized = face_recognized

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()