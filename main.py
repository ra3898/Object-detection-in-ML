# from qiskit import *
# from qiskit import IBMQ
# from qiskit.compiler import transpile, assemble
# from qiskit.tools.jupyter import *
# from qiskit.visualization import *
from io import BytesIO
import streamlit as st
import numpy as np
from PIL import Image, ImageColor
# from streamlit_webrtc import webrtc_streamer, RTCConfiguration
# import av
import cv2
import os
import csv

import time
import pandas as pd
import cv2
import datetime
import streamlit as st
# Importing standard Qiskit libraries and configuring account


import time
import pandas as pd
import cv2
import datetime
import streamlit as st

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')

from PIL import Image
style.use('default')
import numpy as np
import os
import glob
from PIL import Image, ImageFilter
from PIL import Image
import datetime
import time
import time
from numpy import asarray

# Set page configs. Get emoji names from WebFx
st.set_page_config(page_title="Real-time Face Detection", page_icon="./assets/faceman_cropped.png", layout="centered")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; "> Realtime Frontal Face Detection</p>'
st.markdown(title, unsafe_allow_html=True)

# st.markdown(
#     "Frontal-face detection using *Haar-Cascade Algorithm* which is one of the oldest face detection algorithms "
#     "invented. It is based on the sliding window approach,giving a real-time experience with 30-40 FPS on any average CPU.")

# supported_modes = "<html> " \
#                   "<body><div> <b>Supported Face Detection Modes (Change modes from sidebar menu)</b>" \
#                   "<ul><li>Image Upload</li><li>Webcam Image Capture</li><li>Webcam Video Realtime</li></ul>" \
#                   "</div></body></html>"
# st.markdown(supported_modes, unsafe_allow_html=True)

st.warning("NOTE : Click the arrow icon at Top-Left to open Sidebar menu. ")

# -------------Sidebar Section------------------------------------------------

detection_mode = None
# Haar-Cascade Parameters
minimum_neighbors = 4
# Minimum possible object size
min_object_size = (50, 50)
# bounding box thickness
bbox_thickness = 3
# bounding box color
bbox_color = (0, 255, 0)

with st.sidebar:
    st.image("./assets/faceman_cropped.png", width=260)

    title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Face Detection Mode", ('Home','Webcam Image Capture','Webcam Realtime Attendance Fill','Train Faces','Manual Attendance'), index=0)
    if mode == "Home":
        detection_mode = mode
    if mode == "Webcam Image Capture":
        detection_mode = mode
    elif mode == 'Webcam Realtime Attendance Fill':
        detection_mode = mode
    # elif mode == 'capture':
    #     detection_mode = mode
    elif mode == 'Train Faces':
        detection_mode = mode
    elif mode == 'Train with Quantum Image edge Detection':
        detection_mode = mode
    elif mode == 'Manual Attendance':
        detection_mode = mode
    
    # slider for choosing parameter values
    # minimum_neighbors = st.slider("Mininum Neighbors", min_value=0, max_value=10,
    #                               help="Parameter specifying how many neighbors each candidate "
    #                                    "rectangle should have to retain it. This parameter will affect "
    #                                    "the quality of the detected faces. Higher value results in less "
    #                                    "detections but with higher quality.",
    #                               value=minimum_neighbors)

    # slider for choosing parameter values

    # min_size = st.slider(f"Mininum Object Size, Eg-{min_object_size} pixels ", min_value=3, max_value=500,
    #                      help="Minimum possible object size. Objects smaller than that are ignored.",
    #                      value=70)

    # min_object_size = (min_size, min_size)

    # # Get bbox color and convert from hex to rgb
    # bbox_color = ImageColor.getcolor(str(st.color_picker(label="Bounding Box Color", value="#00FF00")), "RGB")

    # # ste bbox thickness
    # bbox_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=30,
    #                            help="Sets the thickness of bounding boxes",
    #                            value=bbox_thickness)

    # st.info("NOTE : The quality of detection will depend on above paramters."
    #         " Try adjusting them as needed to get the most optimal output")

    # line break
    # st.markdown(" ")
# -------------Image Upload Section------------------------------------------------

# if detection_mode == "Image Upload":

#     # Example Images
#     col1, col2 = st.columns(2)
#     with col1:
#         st.image(image="./assets/example_2.png")
#     with col2:
#         st.image(image="./assets/example_3.png")

#     uploaded_file = st.file_uploader("Upload Image (Only PNG & JPG images allowed)", type=['png', 'jpg'])

#     if uploaded_file is not None:

#         with st.spinner("Detecting faces..."):
#             img = Image.open(uploaded_file)

#             # To convert PIL Image to numpy array:
#             img = np.array(img)

#             # Load the cascade
#             face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#             # Convert into grayscale
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             # Detect faces
#             faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=minimum_neighbors, minSize=min_object_size)

#             if len(faces) == 0:
#                 st.warning(
#                     "No Face Detected in Image. Make sure your face is visible in the camera with proper lighting."
#                     " Also try adjusting detection parameters")
#             else:
#                 # Draw rectangle around the faces
#                 for (x, y, w, h) in faces:
#                     cv2.rectangle(img, (x, y), (x + w, y + h), color=bbox_color, thickness=bbox_thickness)

#                 # Display the output
#                 st.image(img)

#                 if len(faces) > 1:
#                     st.success("Total of " + str(
#                         len(faces)) + " faces detected inside the image. Try adjusting minimum object size if we missed anything")

#                     # convert to pillow image
#                     img = Image.fromarray(img)
#                     buffered = BytesIO()
#                     img.save(buffered, format="JPEG")

#                     # Creating columns to center button
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         pass
#                     with col3:
#                         pass
#                     with col2:
#                         st.download_button(
#                             label="Download image",
#                             data=buffered.getvalue(),
#                             file_name="output.png",
#                             mime="image/png")
#                 else:
#                     st.success(
#                         "Only 1 face detected inside the image. Try adjusting minimum object size if we missed anything.")

#                     # convert to pillow image
#                     img = Image.fromarray(img)
#                     buffered = BytesIO()
#                     img.save(buffered, format="JPEG")

#                     # Creating columns to center button
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         pass
#                     with col3:
#                         pass
#                     with col2:
#                         st.download_button(
#                             label="Download image",
#                             data=buffered.getvalue(),
#                             file_name="output.png",
#                             mime="image/png")

#function for webcam
if detection_mode == "Home":
    st.title("Home🏡")

if detection_mode == "Manual Attendance":
    st.title("Fill Attendance Manually")
    enrollmentstu = st.number_input("Enter Enrollment Id",format="%i",step=1)
    namestu= st.text_input("Enter Name")
    datestu = st.date_input("Enter Date")
    substu= st.text_input("Enter Subject")
    if datestu and substu and namestu and enrollmentstu:
        st.text(enrollmentstu)
        st.text(namestu)
        st.text(substu)
        st.text(datestu)
        row = [enrollmentstu, namestu, substu, datestu]
        with open('ManualAttendance\ManualAttendance.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(row)
            csvFile.close()
        st.info("Attendance Filled")

def detect_web(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img=image, pt1=(x, y), pt2=(
            x + w, y + h), color=(255, 0, 0), thickness=2)

    return image, faces

if detection_mode == "capture":
    st.header("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # while run:
    #     # Reading image from video stream
    #     _, img = camera.read()
    #     # Call method we defined above
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ace_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # faces = face_cascade.detectMultiScale(
        #     image=image, scaleFactor=1.3, minNeighbors=5)

        # for (x, y, w, h) in faces:
        #     cv2.rectangle(img=image, pt1=(x, y), pt2=(
        #         x + w, y + h), color=(255, 0, 0), thickness=2)

        # return image, face
    #     img, a = detect_web(img)
    #     # st.image(img, use_column_width=True)
    #     FRAME_WINDOW.image(img)

    # cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    Enrollment = st.text_input("Enter Enrollment ID",value="333")
    Name = st.text_input("Enter Name",value="vivek")
    sampleNum = 0
    while (run):
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite("TrainingImage/ " + Name + "." + Enrollment + '.' + str(sampleNum) + ".jpg",
                        gray)
            print("Images Saved for Enrollment :")
            FRAME_WINDOW.image(img)
            print(sampleNum)

            if sampleNum == 20:
                break
            # cv2.imshow('Frame', img)
        # wait for 100 miliseconds
        st.text(f"Images SAved {sampleNum}")
        if sampleNum == 20:
                break# if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100))  # Resize images to 100x100
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img)
        Ids.append(id)
    return faceSamples, Ids

if detection_mode == "Train Faces":
        
    # Path for face image database
    path = 'TrainingImage'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    st.text("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('TrainingImageLabel/Trainner.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    st.text("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
# -------------Webcam Image Capture Section------------------------------------------------

if detection_mode == "Webcam Image Capture":

    st.info("NOTE : In order to use this mode, you need to give webcam access.")
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    #make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
    face_id= st.number_input('\n Enter User id end press <return>  ',format="%i",min_value=1, max_value=99999999, value=5, step=1)
    # st.text(face_id)
    # st.text(type(face_id))
    name_id = st.text_input('\n Enter Name end press <return>  ')
    if face_id and name_id:
        st.text("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
        count = 0

        #start detect your face and take 30 pictures
        while(True):

            ret, img = cam.read()
            gray = img
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:

                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("TrainingImage/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

                cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30: # Take 30 face sample and stop video
                break

        # Do a bit of cleanup
        st.text("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        Enrollment= face_id
        Name = name_id
        ts = time.time()
        Date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        Time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        row = [Enrollment, Name, Date, Time]
        with open('StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(row)
            csvFile.close()
        st.text(Enrollment)
        st.text(Name)
        res = "Images Saved for Enrollment : " + str(Enrollment) + " Name : " + Name
        st.info(res)
    import pandas as pd
    # folder_name='StudentDetails'
    # csv_file= 'StudentDetails.csv'
    # file_path = os.path.join(folder_name, csv_file)
    capturedf = pd.read_csv(r"StudentDetails/StudentDetails.csv")
    # st.dataframe(capturedf)
    capturedf = capturedf.drop_duplicates(['Enrollment'], keep='first')
    st.dataframe(capturedf)
    capturedf.to_csv(r"StudentDetails/StudentDetails.csv", index=False)
    st.text("Done")  
        
# -------------Webcam Realtime Section------------------------------------------------
def hours_to_timestamp(hours):
    # Parse hours input
    hour, minute, second = map(int, hours.split(':'))

    # Construct datetime object for today with the given hours
    now = datetime.datetime.now()
    desired_time = datetime.datetime(now.year, now.month, now.day, hour, minute, second)

    # Convert the datetime object to a Unix timestamp
    timestamp = time.mktime(desired_time.timetuple())

    return timestamp
# minW = 0.1 * cam.get(3)

import time
import datetime
import cv2
import pandas as pd
import streamlit as st

# if detection_mode == "Webcam Realtime Attendance Fill":
#     subject = st.text_input("Enter Subject")
#     current_time = time.time()

#     # Convert the timestamp to a readable string format
#     current_time_str = time.ctime(current_time)

#     # Display the current time
#     st.info(current_time_str)

#     # Load enrollment details
#     df = pd.read_csv("StudentDetails/StudentDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Enrollment', 'Name', 'Date', 'Time', 'Subject']
#     attendance = pd.DataFrame(columns=col_names)

#     if subject:
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         recognizer.read('TrainingImageLabel/Trainner.yml')  # Load the trained model
#         cascadePath = "haarcascade_frontalface_default.xml"
#         faceCascade = cv2.CascadeClassifier(cascadePath)

#         # Define video frame size
#         cam.set(3, 640)  # Set video width
#         cam.set(4, 480)  # Set video height

#         # Define min window size to be recognized as a face
#         minW = 0.1 * cam.get(3)
#         minH = 0.1 * cam.get(4)

#         # Dictionary to store the time when a face is first detected
#         face_detected_time = {}

#         st.info("Starting to capture images every 10 minutes...")

#         while True:
#             # Capture the current time
#             current_time = time.time()

#             # Capture image every 10 minutes
#             ret, img = cam.read()
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=5,
#                 minSize=(int(minW), int(minH)),
#             )

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

#                 # Check if confidence is less than 100 ==> "0" is perfect match
#                 if confidence < 100:
#                     ts = time.time()
#                     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

#                     # Check if enrollment is present in the CSV file
#                     if id in df['Enrollment'].values:
#                         # Fetch the student name
#                         namesstu = df.loc[df['Enrollment'] == id]['Name'].values
#                         name = namesstu[0] if len(namesstu) > 0 else "Unknown"
#                     else:
#                         # If the enrollment is not found in the CSV
#                         name = "Unknown - Not present in Enrollment CSV"
#                         st.error(f"Person with Enrollment ID {id} is not present in the Enrollment CSV file.")

#                     # If the face is seen for the first time, store the detection time
#                     if id not in face_detected_time:
#                         face_detected_time[id] = ts

#                     # Only add to attendance if the face is detected for 10 seconds
#                     if ts - face_detected_time[id] >= 10:
#                         # Add to attendance
#                         attendance.loc[len(attendance)] = [id, name, date, timeStamp, subject]
#                         st.info(f"Attendance marked for {name}")
#                         del face_detected_time[id]  # Remove the face after attendance is marked
#                 else:
#                     id = "unknown"
#                     confidence = "  {0}%".format(round(100 - confidence))

#                 # Display the image with bounding box and confidence level
#                 cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#                 cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

#             cv2.imshow('camera', img)

#             # Save attendance to CSV after each capture
#             ts = time.time()
#             date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#             timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#             Hour, Minute, Second = timeStamp.split(":")
#             fileName = f"Attendance/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
#             attendance.to_csv(fileName, index=False)

#             st.text(f"Attendance saved successfully for {subject}.")
#             st.text(attendance)

#             # Check for missing students in attendance
#             student_details_df = pd.read_csv("StudentDetails/StudentDetails.csv")
#             attendance_enrollments = attendance['Enrollment'].unique()
#             student_enrollments = student_details_df['Enrollment'].unique()

#             # Find missing enrollments
#             missing_enrollments = set(student_enrollments) - set(attendance_enrollments)

#             # Display the missing enrollments, if any
#             if missing_enrollments:
#                 st.error(f"Missing Enrollment IDs in Attendance: {', '.join(map(str, missing_enrollments))}")
#             else:
#                 st.success("All students are present in the attendance.")

#             # Wait for 10 minutes (600 seconds)
#             time.sleep(600)

#             # Press 'ESC' to exit the loop
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break

#         cam.release()
#         cv2.destroyAllWindows()
if detection_mode == "Webcam Realtime Attendance Fill":
    subject = st.text_input("Enter Subject")
    current_time = time.time()
    

    # Convert the timestamp to a readable string format
    current_time_str = time.ctime(current_time)

    # Display the current time
    st.info(current_time_str)

    # Load enrollment details
    df = pd.read_csv("StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Enrollment', 'Name', 'Date', 'Time', 'Subject']
    attendance = pd.DataFrame(columns=col_names)

    if subject:
        st.info("Capturing Image")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('TrainingImageLabel/Trainner.yml')  # load trained model
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath)

        # Initialize real-time video capture
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        # Dictionary to store the time when a face is first detected
        face_detected_time = {}

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less than 100 ==> "0" is perfect match
                if confidence < 100:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                    # Check if enrollment is present in the csv file
                    if id in df['Enrollment'].values:
                        # Fetch the student name
                        namesstu = df.loc[df['Enrollment'] == id]['Name'].values
                        name = namesstu[0] if len(namesstu) > 0 else "Unknown"
                    else:
                        # If the enrollment is not found in the CSV
                        name = "Unknown - Not present in Enrollment CSV"
                        st.error(f"Person with Enrollment ID {id} is not present in the Enrollment CSV file.")

                    # If the face is seen for the first time, store the detection time
                    if id not in face_detected_time:
                        face_detected_time[id] = ts

                    # Check if the face has been detected for at least 10 seconds
                    if ts - face_detected_time[id] >= 10:
                        # Only add to attendance if the face is detected for 10 seconds
                        attendance.loc[len(attendance)] = [id, name, date, timeStamp, subject]
                        st.info(f"Attendance marked for {name}")
                        del face_detected_time[id]  # Remove the face after attendance is marked
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            # Press 'ESC' to exit video
            
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Save attendance to CSV
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        fileName = f"Attendance/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
        attendance.to_csv(fileName, index=False)
        st.text(type(attendance))
        st.text("Attendance saved successfully.")
        st.text(attendance)
        student_details_df = pd.read_csv("StudentDetails/StudentDetails.csv")
        # Get the list of enrollments from both dataframes
        attendance_enrollments = attendance['Enrollment'].unique()
        student_enrollments = student_details_df['Enrollment'].unique()

        # Find missing enrollments
        missing_enrollments = set(student_enrollments) - set(attendance_enrollments)

        # Display the missing enrollments, if any
        if missing_enrollments:
            st.error(f"Missing Enrollment IDs in Attendance: {', '.join(map(str, missing_enrollments))}")
        else:
            st.success("All students are present in the attendance.")

# if detection_mode == "Webcam Realtime Attendance Fill":
#     subject = st.text_input("Enter Subject")
#     current_time = time.time()

#     # Convert the timestamp to a readable string format
#     current_time_str = time.ctime(current_time)

#     # Display the current time
#     st.info(current_time_str)

#     # Load enrollment details
#     df = pd.read_csv("StudentDetails/StudentDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Enrollment', 'Name', 'Date', 'Time', 'Subject']
#     attendance = pd.DataFrame(columns=col_names)

#     if subject:
#         st.info("Capturing Image")
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         recognizer.read('TrainingImageLabel/Trainner.yml')  # load trained model
#         cascadePath = "haarcascade_frontalface_default.xml"
#         faceCascade = cv2.CascadeClassifier(cascadePath)

#         # Initialize real-time video capture
#         cam.set(3, 640)  # set video width
#         cam.set(4, 480)  # set video height

#         # Define min window size to be recognized as a face
#         minW = 0.1 * cam.get(3)
#         minH = 0.1 * cam.get(4)

#         # Dictionary to store the time when a face is first detected
#         face_detected_time = {}

#         while True:
#             ret, img = cam.read()
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=5,
#                 minSize=(int(minW), int(minH)),
#             )

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

#                 # Check if confidence is less than 100 ==> "0" is perfect match
#                 if confidence < 100:
#                     ts = time.time()
#                     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

#                     # Check if enrollment is present in the csv file
#                     if id in df['Enrollment'].values:
#                         # Fetch the student name
#                         namesstu = df.loc[df['Enrollment'] == id]['Name'].values
#                         name = namesstu[0] if len(namesstu) > 0 else "Unknown"
#                     else:
#                         # If the enrollment is not found in the CSV
#                         name = "Unknown - Not present in Enrollment CSV"
#                         st.error(f"Person with Enrollment ID {id} is not present in the Enrollment CSV file.")

#                     # If the face is seen for the first time, store the detection time
#                     if id not in face_detected_time:
#                         face_detected_time[id] = ts

#                     # Check if the face has been detected for at least 10 seconds
#                     if ts - face_detected_time[id] >= 10:
#                         # Only add to attendance if the face is detected for 10 seconds
#                         attendance.loc[len(attendance)] = [id, name, date, timeStamp, subject]
#                         st.info(f"Attendance marked for {name}")
#                         del face_detected_time[id]  # Remove the face after attendance is marked
#                 else:
#                     id = "unknown"
#                     confidence = "  {0}%".format(round(100 - confidence))

#                 cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#                 cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

#             cv2.imshow('camera', img)

#             # Press 'ESC' to exit video
#             k = cv2.waitKey(10) & 0xff
#             if k == 27:
#                 break

#         cam.release()
#         cv2.destroyAllWindows()

#         # Save attendance to CSV
#         ts = time.time()
#         date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#         timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#         Hour, Minute, Second = timeStamp.split(":")
#         fileName = f"Attendance/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
#         attendance.to_csv(fileName, index=False)
#         st.text(type(attendance))
#         st.text("Attendance saved successfully.")
#         st.text(attendance)
#         student_details_df = pd.read_csv("StudentDetails/StudentDetails.csv")
#         # Get the list of enrollments from both dataframes
#         attendance_enrollments = attendance['Enrollment'].unique()
#         student_enrollments = student_details_df['Enrollment'].unique()

#         # Find missing enrollments
#         missing_enrollments = set(student_enrollments) - set(attendance_enrollments)

#         # Display the missing enrollments, if any
#         if missing_enrollments:
#             st.error(f"Missing Enrollment IDs in Attendance: {', '.join(map(str, missing_enrollments))}")
#         else:
#             st.success("All students are present in the attendance.")

# if detection_mode == "Webcam Realtime Attendance Fill":
#     subject = st.text_input("Enter Subject")
#     current_time = time.time()

#     # Convert the timestamp to a readable string format
#     current_time_str = time.ctime(current_time)

#     # Display the current time
#     st.info(current_time_str)

#     df = pd.read_csv("StudentDetails/StudentDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Enrollment', 'Name', 'Date', 'Time', 'Subject']
#     attendance = pd.DataFrame(columns=col_names)

#     if subject:
#         st.info("Capturing Image")
#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         recognizer.read('TrainingImageLabel/Trainner.yml')  # load trained model
#         cascadePath = "haarcascade_frontalface_default.xml"
#         faceCascade = cv2.CascadeClassifier(cascadePath)

#         # Initialize real-time video capture
#         cam.set(3, 640)  # set video width
#         cam.set(4, 480)  # set video height

#         # Define min window size to be recognized as a face
#         minW = 0.1 * cam.get(3)
#         minH = 0.1 * cam.get(4)

#         # Dictionary to store the time when a face is first detected
#         face_detected_time = {}

#         while True:
#             ret, img = cam.read()
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.2,
#                 minNeighbors=5,
#                 minSize=(int(minW), int(minH)),
#             )

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

#                 # Check if confidence is less than 100 ==> "0" is perfect match
#                 if confidence < 100:
#                     ts = time.time()
#                     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

#                     namesstu = (df.loc[df['Enrollment'] == id]['Name'].values)
#                     name = namesstu[0] if len(namesstu) > 0 else "Unknown"

#                     # If the face is seen for the first time, store the detection time
#                     if id not in face_detected_time:
#                         face_detected_time[id] = ts

#                     # Check if the face has been detected for at least 10 seconds
#                     if ts - face_detected_time[id] >= 10:
#                         # Only add to attendance if the face is detected for 10 seconds
#                         attendance.loc[len(attendance)] = [id, name, date, timeStamp, subject]
#                         st.info(f"Attendance marked for {name}")
#                         del face_detected_time[id]  # Remove the face after attendance is marked
#                 else:
#                     id = "unknown"
#                     confidence = "  {0}%".format(round(100 - confidence))

#                 cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#                 cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

#             cv2.imshow('camera', img)

#             # Press 'ESC' to exit video
#             k = cv2.waitKey(10) & 0xff
#             if k == 27:
#                 break

#         cam.release()
#         cv2.destroyAllWindows()

#         # Save attendance to CSV
#         ts = time.time()
#         date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#         timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#         Hour, Minute, Second = timeStamp.split(":")
#         fileName = f"Attendance/{subject}_{date}_{Hour}-{Minute}-{Second}.csv"
#         attendance.to_csv(fileName, index=False)

#         st.text("Attendance saved successfully.")
#         st.text(attendance)

# minH = 0.1 * cam.get(4)
# if detection_mode == "Webcam Realtime Attendance Fill":
#     subject=st.text_input("Enter Subject")
#     current_time = time.time()

#     # Convert the timestamp to a readable string format
#     current_time_str = time.ctime(current_time)

#     # Display the current time
#     st.info(current_time_str)
#     import pandas as pd
#     df = pd.read_csv("StudentDetails\StudentDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     col_names = ['Enrollment', 'Name', 'Date', 'Time','Subject']
#     attendance = pd.DataFrame(columns=col_names)
#     # subtime = st.time_input("Set an alarm for", datetime.time(8, 45))
#     if subject:
#             st.info("Capturing Image")
#             recognizer = cv2.face.LBPHFaceRecognizer_create()
#             recognizer.read('TrainingImageLabel/Trainner.yml')  # load trained model
#             cascadePath = "haarcascade_frontalface_default.xml"
#             faceCascade = cv2.CascadeClassifier(cascadePath)

#             font = cv2.FONT_HERSHEY_SIMPLEX

#             # iniciate id counter, the number of persons you want to include
#             id = 1  # two persons (e.g. Jacob, Jack)

#             names = ['','333']  # key in names, start from the second place, leave first empty

#             # Initialize and start realtime video capture
#             cam = cv2.VideoCapture(0)
#             cam.set(3, 640)  # set video widht
#             cam.set(4, 480)  # set video height

#             # Define min window size to be recognized as a face
#             minW = 0.1 * cam.get(3)
#             minH = 0.1 * cam.get(4)

#             while True:

#                 ret, img = cam.read()
#                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 faces = faceCascade.detectMultiScale(
#                     gray,
#                     scaleFactor=1.2,
#                     minNeighbors=5,
#                     minSize=(int(minW), int(minH)),
#                 )
#                 lis = []
#                 for (x, y, w, h) in faces:

#                     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

#                     # Check if confidence is less them 100 ==> "0" is perfect match
#                     if (confidence < 100):
#                         global Subject
#                         global aa
#                         global date
#                         global timeStamp
#                         # Subject = tx.get()
#                         ts = time.time()
#                         date = datetime.datetime.fromtimestamp(
#                             ts).strftime('%Y-%m-%d')
#                         timeStamp = datetime.datetime.fromtimestamp(
#                             ts).strftime('%H:%M:%S')
#                         # st.dataframe(df)
#                         namesstu= (df.loc[df['Enrollment']==1]['Name'].values)
#                         # st.text(namesstu[0])
#                         # aa = df.loc[df['Enrollment'] == 1]['Name'].values
#                         global tt
#                         tt = str(id) + "-" + namesstu[0]
#                         # En = '15624031' + str(id)
#                         # st.dataframe(attendance)
#                         attendance.loc[len(attendance)] = [
#                             id, namesstu[0], date, timeStamp,subject]
#                         print(id)
#                         lis.append(id)
#                         id = names[id]
#                         confidence = "  {0}%".format(round(100 - confidence))
#                     else:
#                         id = "unknown"
#                         confidence = "  {0}%".format(round(100 - confidence))

#                     cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
#                     cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
#                     # lis.append(id)
#                 cv2.imshow('camera', img)
#                 # cv2.imshow('Filling attedance..', im)
#                 k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
#                 if k == 27:
#                     break
#                 attendance = attendance.drop_duplicates(
#                         ['Enrollment'], keep='first')
            
#             cam.release()
#             cv2.destroyAllWindows()
#             # Do a bit of cleanup
#             st.text(lis)
#             ts = time.time()
#             date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#             timeStamp = datetime.datetime.fromtimestamp(
#                 ts).strftime('%H:%M:%S')
#             Hour, Minute, Second = timeStamp.split(":")
#             fileName = "Attendance/" + subject + "_" + date + \
#                 "_" + Hour + "-" + Minute + "-" + Second + ".csv"
#             attendance = attendance.drop_duplicates(
#                 ['Enrollment'], keep='first')
#             st.text(attendance)
#             attendance.to_csv(fileName, index=False)
#             fileName = subject + "_" + date + \
#                 "_" + Hour + "-" + Minute + "-" + Second + ".csv"
#             # st.dataframe(fileName)
#             print("\n [INFO] Exiting Program and cleanup stuff")
            


# # # if detection_mode == "Train with Quantum Image edge Detection":
# # #     if st.button("Train Images with Quantum image Edge detection"):
# # #         # Convert the raw pixel values to probability amplitudes
# # #         def amplitude_encode(img_data):

# # #             # Calculate the RMS value
# # #             rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))

# # #             # Create normalized image
# # #             image_norm = []
# # #             for arr in img_data:
# # #                 for ele in arr:
# # #                     if rms==0:
# # #                         image_norm.append(0)
# # #                     else:
# # #                         image_norm.append(ele / rms)

# # #             # Return the normalized image as a numpy array
# # #             return np.array(image_norm)



# # #         def train_quant(imagepath):

# # #             image_size = 256       # Original image-width
# # #             image_crop_size = 8   # Width of each part of image for processing


# # #             # Load the image from filesystem
# # #             image_raw = np.array(Image.open(imagepath))
# # #             print('Raw Image info:', image_raw.shape)
# # #             print('Raw Image datatype:', image_raw.dtype)

# # #             # Convert the RBG component of the image to B&W image, as a numpy (uint8) array
# # #             image = []
# # #             for i in range(image_size):
# # #                 image.append([])
# # #                 for j in range(image_size):
# # #                     image[i].append(image_raw[i][j][0] / 255)

# # #             image = np.array(image)
# # #             print('Image shape (numpy array):', image.shape)
# # #             # # Display the image
# # #             # plt.title('Big Image')
# # #             # plt.xticks(range(0, image.shape[0]+1, 32))
# # #             # plt.yticks(range(0, image.shape[1]+1, 32))
# # #             # plt.imshow(image, extent=[0, image.shape[0], image.shape[1], 0], cmap='viridis')
# # #             # plt.show()
# # #             # Initialize some global variable for number of qubits
# # #             data_qb = 6
# # #             anc_qb = 1
# # #             total_qb = data_qb + anc_qb

# # #             # Initialize the amplitude permutation unitary
# # #             D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
# # #             crops=[]
# # #             for i in range(image_size//image_crop_size):
# # #                 for j in range(image_size//image_crop_size):
# # #                     crops.append(image[i*image_crop_size:(i+1)*image_crop_size,j*image_crop_size:(j+1)*image_crop_size])
            
# # #             # Get the amplitude ancoded pixel values
# # #             # Horizontal: Original image
# # #             image_norm_h = amplitude_encode(image)

# # #             # Vertical: Transpose of Original image
# # #             image_norm_v = amplitude_encode(image.T)

# # #             edge_crops=[]
# # #             x=0
# # #             for crop in crops:
# # #                 if not((crop==0).all() or (crop==1).all()):
# # #                     # Horizontal: Original image
# # #                     image_norm_h = amplitude_encode(crop)

# # #                     # Vertical: Transpose of Original image
# # #                     image_norm_v = amplitude_encode(crop.T)
# # #                     # Create the circuit for horizontal scan
# # #                     qc_h = QuantumCircuit(total_qb)
# # #                     qc_h.initialize(image_norm_h, range(1, total_qb))
# # #                     qc_h.h(0)
# # #                     qc_h.unitary(D2n_1, range(total_qb))
# # #                     qc_h.h(0)

# # #                     # Create the circuit for vertical scan
# # #                     qc_v = QuantumCircuit(total_qb)
# # #                     qc_v.initialize(image_norm_v, range(1, total_qb))
# # #                     qc_v.h(0)
# # #                     qc_v.unitary(D2n_1, range(total_qb))
# # #                     qc_v.h(0)

# # #                     # Combine both circuits into a single list
# # #                     circ_list = [qc_h, qc_v]

# # #                     # Simulating the cirucits
# # #                     back = Aer.get_backend('statevector_simulator')
# # #                     results = execute(circ_list, backend=back).result()
# # #                     sv_h = results.get_statevector(qc_h)
# # #                     sv_v = results.get_statevector(qc_v)

# # #                     threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

# # #                     # Selecting odd states from the raw statevector and
# # #                     # reshaping column vector of size 64 to an 8x8 matrix
# # #                     edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(8, 8)
# # #                     edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(8, 8).T
# # #                     edge_scan_sim = edge_scan_h | edge_scan_v

# # #                     edge_crops.append(edge_scan_sim)
# # #                 else:
# # #                     edge_crops.append(crop)

# # #                 if x%32==0:
# # #                     print(x)
# # #                 x+=1
# # #             tmps=[]
# # #             for j in range(32):
# # #                 init=edge_crops[32*j]
# # #                 for i in range(1,32):
# # #                     init=np.concatenate((init, edge_crops[32*j+i]), axis=1)
# # #                 tmps.append(init)
# # #             actual_edge_image=tmps[0]
# # #             for i in range(1,32):
# # #                 actual_edge_image=np.concatenate((actual_edge_image, tmps[i]), axis=0)
# # #             return actual_edge_image
        
# # #         def process_images(input_folder, output_folder):
# # #             st.text("procss_image")
# # #             # Create the output folder if it doesn't exist
# # #             os.makedirs(output_folder, exist_ok=True)
            
# # #             # Get a list of all image files in the input folder
# # #             image_paths = glob.glob(os.path.join(input_folder, '*'))
            
# # #             # Process each image
# # #             for image_path in image_paths:
# # #                 try:
# # #                     print(image_path)
# # #                     actual_edge_image= train_quant(image_path)
                    
# # #                     # Generate the output file path
# # #                     base_name = os.path.basename(image_path)
# # #                     output_path = os.path.join(output_folder, base_name)
# # #                     print(output_path)

# # #                     data = Image.fromarray(actual_edge_image)
# # #                     plt.imsave(output_path, data)
                    
# # #                     # Save the processed image
# # #                     # image.save(output_path)
                    
# # #                     # print(f"Processed and saved: {output_path}")
# # #                 except Exception as e:
# # #                     # None
# # #                     print(f"Failed to process {image_path}: {e}")

# #         # Example usage
# #         input_folder = r"D:\Viren\Qriocity\facerecc (2)\Realtime-Face-Detection\TrainingImage"
# #         output_folder = r"D:\Viren\Qriocity\facerecc (2)\Realtime-Face-Detection\TrainingImage\outputfolder"
# #         process_images(input_folder, output_folder)

#         path = 'TrainingImage\outputfolder'

#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

#         # function to get the images and label data
#         def getImagesAndLabels(path):

#             imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
#             faceSamples=[]
#             ids = []

#             for imagePath in imagePaths:

#                 PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
#                 img_numpy = np.array(PIL_img,'uint8')

#                 id = int(os.path.split(imagePath)[-1].split(".")[1])
#                 faces = detector.detectMultiScale(img_numpy)

#                 for (x,y,w,h) in faces:
#                     faceSamples.append(img_numpy[y:y+h,x:x+w])
#                     ids.append(id)

#             return faceSamples,ids

#         st.text("\n [INFO] Training faces. It will take a few seconds. Wait ...")
#         faces,ids = getImagesAndLabels(path)
#         recognizer.train(faces, np.array(ids))

#         # Save the model into trainer/trainer.yml
#         recognizer.write('TrainingImageLabel/Trainner.yml') # recognizer.save() worked on Mac, but not on Pi

#         # Print the numer of faces trained and end program
#         st.text("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
# -------------Hide Streamlit Watermark------------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
