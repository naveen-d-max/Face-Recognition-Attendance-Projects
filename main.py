import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(f"Found images: {myList}")

# Load training images
for cl in myList:
    try:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is None:
            print(f"Warning: Could not load image {cl}")
            continue
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    except Exception as e:
        print(f"Error loading {cl}: {e}")

print(f"Loaded class names: {classNames}")


def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            if encode:
                encodeList.append(encode[0])
            else:
                print(f"Warning: No face found in image {idx}")
        except Exception as e:
            print(f"Error encoding image {idx}: {e}")
    return encodeList


def markAttendance(name):
    try:
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()

            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                print(f"Attendance marked for {name}")
    except FileNotFoundError:
        print("Creating new Attendance.csv file")
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time')
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')
    except Exception as e:
        print(f"Error marking attendance: {e}")


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

if not images:
    print("Error: No training images found! Please add images to Training_images folder.")
    exit(1)

encodeListKnown = findEncodings(images)
print(f'Encoding Complete. Encoded {len(encodeListKnown)} faces')

if not encodeListKnown:
    print("Error: No valid encodings found! Please check your training images.")
    exit(1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

print("Starting face recognition. Press 'q' to quit.")

while True:
    success, img = cap.read()
    
    if not success:
        print("Error: Failed to read frame from webcam")
        break
    
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    try:
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
    except Exception as e:
        print(f"Error in face recognition loop: {e}")

    cv2.imshow('Webcam', img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program ended")
