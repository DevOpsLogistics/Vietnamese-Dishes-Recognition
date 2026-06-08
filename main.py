import os
import pickle
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import cosine
from deepface import DeepFace

# Hỗ trợ đọc ảnh HEIC từ iPhone (nếu có)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DATASET_FOLDER = "./dataset" # Thư mục chứa các thư mục con (mỗi thư mục là 1 người)
DB_PATH = "./face_database.pkl"
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68 # Ngưỡng chuẩn của mô hình ArcFace là 0.68

def build_database():
    print(f"\n Đang kiểm tra cơ sở dữ liệu tại '{DB_PATH}'...")
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            face_db = pickle.load(f)
        print(f" Đã tải thành công {len(face_db)} khuôn mặt từ database có sẵn.")
        return face_db
        
    print(" Chưa có Database. Bắt đầu đọc ảnh từ thư mục và phân tích...")
    print(f" Thư mục chứa ảnh: {os.path.abspath(DATASET_FOLDER)}")
    
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
        print(f" Lỗi: Không tìm thấy thư mục '{DATASET_FOLDER}'. Mình vừa tạo thư mục này, bạn hãy copy các thư mục người vào trong '{DATASET_FOLDER}' rồi chạy lại nhé!")
        return []
        
    face_db = []
    
    # Duyệt qua từng thư mục của mỗi người
    for person_name in sorted(os.listdir(DATASET_FOLDER)):
        person_dir = os.path.join(DATASET_FOLDER, person_name)
        if not os.path.isdir(person_dir): 
            continue
            
        success, fail = 0, 0
        for img_file in os.listdir(person_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif', '.jfif')): 
                continue
                
            img_path = os.path.join(person_dir, img_file)
            try:
                # Dùng PIL đọc để tránh lỗi đường dẫn Tiếng Việt trên Windows
                img = Image.open(img_path).convert('RGB')
                img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                # Trích xuất đặc trưng khuôn mặt (Embedding)
                result = DeepFace.represent(
                    img_path=img_arr,
                    model_name=MODEL_NAME,
                    enforce_detection=False,
                    detector_backend="mtcnn"
                )
                
                if result and len(result) > 0:
                    face_db.append({'name': person_name, 'embedding': np.array(result[0]['embedding'])})
                    success += 1
                else: 
                    fail += 1
            except Exception as e:
                fail += 1
        
        print(f"   {person_name}: {success} mặt" + (f" (lỗi đọc {fail})" if fail > 0 else ""))

    if not face_db:
        return []

    # Lưu lại DB vào file
    with open(DB_PATH, 'wb') as f:
        pickle.dump(face_db, f)
        
    print(f"\n HOÀN TẤT! Đã lưu vĩnh viễn {len(face_db)} khuôn mặt vào '{DB_PATH}'.")
    return face_db

def recognize_face(img_path, db):
    try:
        # Dùng PIL đọc ảnh để tránh lỗi file/đường dẫn tiếng Việt
        img = Image.open(img_path).convert('RGB')
        img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        result = DeepFace.represent(
            img_path=img_arr,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="mtcnn"
        )
        if not result: 
            return "Không tìm thấy mặt!", 0, []
        
        query_emb = np.array(result[0]['embedding'])
        
        # Tính khoảng cách Cosine với tất cả khuôn mặt trong DB
        distances = [{'name': item['name'], 'distance': cosine(query_emb, item['embedding'])} for item in db]
        distances.sort(key=lambda x: x['distance'])
        
        best = distances[0]
        conf = (1 - best['distance']) * 100
        top5 = distances[:5]
        
        if best['distance'] > THRESHOLD:
            return "Người lạ (không có trong CSDL)", conf, top5
        return best['name'], conf, top5
    except Exception as e:
        return f"Lỗi: {e}", 0, []

if __name__ == "__main__":
    print("="*50)
    print(" HỆ THỐNG NHẬN DIỆN KHUÔN MẶT - CHẠY LOCAL")
    print("="*50)
    
    # 1. Khởi tạo hoặc Tải Database
    face_db = build_database()
    
    if not face_db:
        print(" Dữ liệu trống. Hãy thêm các thư mục ảnh vào trong thư mục 'dataset' và chạy lại script.")
        exit()
        
    # 2. Vòng lặp nhận diện liên tục
    while True:
        print("\n" + "-"*50)
        print(" HƯỚNG DẪN: Kéo thả file ảnh cần nhận diện từ thư mục thả trực tiếp vào cửa sổ này (hoặc gõ 'exit' để thoát).")
        img_test = input(" Kéo thả file ảnh vào đây: ").strip()
        
        if img_test.lower() == 'exit':
            break
            
        # Xóa dấu nháy kép hoặc nháy đơn nếu kéo thả trên Windows bị dính
        img_test = img_test.strip('"').strip("'")
            
        if not os.path.exists(img_test):
            print(" Không tìm thấy file ảnh. Vui lòng kiểm tra lại đường dẫn!")
            continue
            
        print("\n Đang phân tích khuôn mặt...")
        name, conf, top5 = recognize_face(img_test, face_db)
        
        print(f"\n{'='*40}")
        print(f" KẾT QUẢ: 【{name}】")
        print(f"➤ Độ chính xác: {conf:.1f}%")
        print(f"{'='*40}")
        
        if top5:
            print(f"\n TOP 5 GẦN GIỐNG NHẤT:")
            for i, item in enumerate(top5):
                sim = (1 - item['distance']) * 100
                # Vẽ thanh tiến trình bar code
                bar = '█' * int(sim / 5) + '░' * (20 - int(sim / 5))
                print(f"  {i+1}. {item['name']}: {sim:.1f}% |{bar}|")
