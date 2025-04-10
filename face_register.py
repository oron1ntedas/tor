import cv2
import mediapipe as mp
import numpy as np
import os
import pickle

def main():
    # Получаем путь к директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Создаем путь к папке face_data относительно директории скрипта
    face_data_dir = os.path.join(script_dir, 'face_data')
    
    # Создаем директорию для сохранения данных, если её нет
    if not os.path.exists(face_data_dir):
        os.makedirs(face_data_dir)
        print(f"✅ Создана директория для сохранения образцов: {face_data_dir}")
    
    # Инициализация MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    
    print("📷 Регистрация лица: нажмите 's' для сохранения образца, 'q' для выхода.")
    name = input("Введите имя: ")
    
    sample_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения камеры")
            break
            
        # Конвертируем в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Рисуем точки на лице
                for landmark in landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
        
        # Отображаем количество сохраненных образцов
        cv2.putText(frame, 
                   f"Сохранено образцов: {sample_count}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (255, 255, 255), 
                   2)
        
        cv2.imshow('Face Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if results.multi_face_landmarks:
                # Сохраняем параметры лица
                face_data = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    face_data.append([landmark.x, landmark.y, landmark.z])
                
                # Сохраняем в файл с номером образца
                sample_count += 1
                file_name = f"{name}_{sample_count}.pkl"
                file_path = os.path.join(face_data_dir, file_name)
                with open(file_path, 'wb') as f:
                    pickle.dump(face_data, f)
                print(f"✅ Образец {sample_count} для {name} успешно сохранен в {file_path}")
            else:
                print("❌ Лицо не обнаружено!")
        elif key == ord('q'):
            if sample_count > 0:
                print(f"✅ Регистрация завершена. Сохранено {sample_count} образцов в {face_data_dir}")
            else:
                print("❌ Регистрация отменена")
            break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
