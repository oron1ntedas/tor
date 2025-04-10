import os
import urllib.request
import sys

def download_file(url, path):
    try:
        print(f"⬇ Скачивание: {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ Успешно сохранено: {path}")
    except Exception as e:
        print(f"❌ Ошибка при скачивании {url}: {e}")
        sys.exit(1)

def main():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    files = {
        "https://raw.githubusercontent.com/sthanhng/yoloface/master/model-weights/yolov3-face.cfg":
            os.path.join(models_dir, "yolov3-face.cfg"),
        "https://github.com/sthanhng/yoloface/raw/master/model-weights/yolov3-wider_16000.weights":
            os.path.join(models_dir, "yolov3-wider_16000.weights"),
    }

    for url, path in files.items():
        if not os.path.exists(path):
            download_file(url, path)
        else:
            print(f"📁 Уже существует: {path}")

    print("🎉 Все файлы модели скачаны или уже есть!")

if __name__ == "__main__":
    main()
