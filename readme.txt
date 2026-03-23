face_recognition_ilkyar.py dosyasındaki FaceRecognitionSystem isimli class 
insightface kullanarak çalışır. 

model_name='buffalo_l' içerisinde face recognition da bulunan çeşitli 
Computer Vision görevlerini yapabilir.
Daha küçük bir model istenirse 'buffalo_s' şeklinde başlatmak gerekir.

cuda kullanımı için ctx_id = 0, cpu kullanımı için ctx_id = -1 şekil ayarlamak gerekiyor.

fr = FaceRecognitionSystem()

# ilk defa bir yüz tanıtalacaksa 
fr.register_face("Ali Desidero 1", "path/to/ali_1.jpeg")
fr.register_face("Ali Desidero 2", "path/to/ali_2.jpeg")
fr.register_face("Ali Desidero 3", "path/to/ali_3.jpeg")

# save_faces komutu objede kaydedilmiş self.known_faces dict'i kaydetmek için kullanılır. 
# yüzler pickle formatında kaydediliyor
# file_path verilmezse dosyanın olduğu yere kaydediyor
fr.save_faces()

# save_faces metoduyla aynı şekilde çalışıyor
fr.load_faces()


# webcam ile örnek kullanım
fr = FaceRecognitionSystem()
#fr.register_face("Ali Desidero", "/Users/desidero/Desktop/Dersler/diğer/ali.jpg")
fr.load_faces("known_faces.pkl")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

print("Camera initialized successfully. Press 'q' to quit, 'r' to register a new face.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] frame can't be read")
        break

    result_frame, results = fr.recognize_and_draw(frame)
    cv2.putText(result_frame, f"Faces: {len(results)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Recognition", result_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Register new face
        name = input("Enter name for this face: ")
        if fr.register_face(name, frame):
            print(f"Registered {name}")

cap.release()
cv2.destroyAllWindows()