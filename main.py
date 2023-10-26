import cv2
import uuid
import os
import easyocr

car_number_plate_cascade = cv2.CascadeClassifier('model/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture(0)
save_dir = 'car_number_plate_images'

# Perform OCR
def readImages(fileDir='car_number_plate_images'):

    number_plat_images = os.listdir(fileDir)
    reader = easyocr.Reader(['en'], gpu=False)
    
    for number_plat_image in number_plat_images:
        IMG_PATH = os.path.join(fileDir, number_plat_image)
        car_numbers = reader.readtext(IMG_PATH)

        for car_number in car_numbers:
            text = car_number[1]
            with open('car_numbers.txt','a') as f:
                f.write(text+'\n')

# Capture Image From Webcam
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Image Processing 
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_number_plates = car_number_plate_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for (x,y,w,h) in car_number_plates:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0), 5)

        if cv2.waitKey(1) & 0xFF == ord('s'):

            # Crop the car number plate region from the video frame
            car_number_plate_image = frame[y:y + h, x:x + w]

            # Save the car number plate image to a disk
            cv2.imwrite(os.path.join(save_dir, '{}.png'.format(uuid.uuid4())), car_number_plate_image)


    cv2.imshow('Real Time Car Number Plate Extraction', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        readImages()
        break    

# Release resources
cap.release()
cv2.destroyAllWindows()