import os
import cv2
import uuid
from datetime import datetime

class ImageCapture:
    def __init__(self, save_dir='new_user_data', min_images=300):
        self.save_dir = save_dir
        self.min_images = min_images
        os.makedirs(save_dir, exist_ok=True)

    def capture_images(self, label):
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Initialize face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        image_count = 0
        start_time = datetime.now()

        while image_count < self.min_images:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Crop face
                face_img = frame[y:y+h, x:x+w]
                
                # Save face image
                filename = os.path.join(self.save_dir, f'{label}_{uuid.uuid4()}.jpg')
                cv2.imwrite(filename, face_img)
                
                image_count += 1
                
                # Display progress
                cv2.putText(frame, 
                            f'Images Captured: {image_count}/{self.min_images}', 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2)

            cv2.imshow('Image Capture', frame)

            # Break conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or image_count >= self.min_images:
                break

            # Optional: Break if capture takes too long (e.g., 5 minutes)
            if (datetime.now() - start_time).total_seconds() > 300:
                print("Capture timeout reached")
                break

        cap.release()
        cv2.destroyAllWindows()

        return image_count >= self.min_images

def main():
    label = input("Enter new user label: ")
    capturer = ImageCapture()
    success = capturer.capture_images(label)
    
    if success:
        print(f"Successfully captured {capturer.min_images} images for {label}")
    else:
        print("Image capture failed or incomplete")

if __name__ == "__main__":
    main()