# funprojects
learning new things about python by doing some very simple basic projects
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known faces and encodings

# atharv_image = face_recognition.load_image_file("face/atharv.jpg")
# atharv_encoding = face_recognition.face_encodings(atharv_image)[0]

drsonal_image = face_recognition.load_image_file("face/drsonal.jpg")
drsonal_encoding = face_recognition.face_encodings(drsonal_image)[0]

# Prabhanshu_image = face_recognition.load_image_file("face/Prabhanshu.jpg")
# Prabhanshu_encoding = face_recognition.face_encodings(Prabhanshu_image)[0]


mayur_image = face_recognition.load_image_file("face/WhatsApp Image for python.jpg")
mayur_encoding = face_recognition.face_encodings(mayur_image)[0]

# yogesh_image = face_recognition.load_image_file("face/yogesh.jpg")
# yogesh_encoding = face_recognition.face_encodings(yogesh_image)[0]


known_face_encodings = [drsonal_encoding,mayur_encoding]
known_face_names = ["drsonal","Mayur"]

# Create a list to keep track of recognized students
recognized_students = []

# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")


# Open the CSV file for writing attendance data
with open(f"{current_date}.csv", "a", newline="") as f:
    lnwriter = csv.writer(f)

    # Open the video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name not in recognized_students:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 100)
                    fontScale = 1.5
                    fontColor = (255, 0, 0)
                    thickness = 3
                    lineType = 2
                    cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                    recognized_students.append(name)

                    current_time = datetime.now().strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the OpenCV window
    video_capture.release()
    cv2.destroyAllWindows()
