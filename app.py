import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

#### Defining Flask App
app = Flask(__name__)

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise Exception("Could not open video device")

# Set the video frame width and height
cap.set(3, 640)
cap.set(4, 480)

# Loop to capture video frames
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if a frame was read correctly
    if not ret:
        break

    # Display the video frame
    cv2.imshow("Webcam", frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


#### If these directories don't exist, create them

if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/',methods=['GET'])
def home():
    return render_template('register.html')


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    identified_person=""
    newusername=request.args.get('newusername')
    newuserid=request.args.get('newuserid')
    final=newusername+'_'+newuserid

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('register.html', totalreg=totalreg(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) != 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]


            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)


    return redirect(url_for('dashboard', final=final, identified_person=identified_person))


@app.route('/dashboard', methods=['GET'])
def dashboard():
    final=request.args.get('final')
    identified_person=request.args.get('identified_person')
    if final==identified_person:
        return render_template('dashboard.html')
    else:
        return render_template('login.html', mess='Face not recognized. Please try again.')

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/services',methods=['GET'])
def services():
    return render_template('services.html')

@app.route('/contact',methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/logout',methods=['GET'])
def logout():
    return render_template('register.html')



@app.route('/login', methods=['GET'])
def login():
    return render_template('login.html', totalreg=totalreg())

#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return render_template('register.html')


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True,port=8000)