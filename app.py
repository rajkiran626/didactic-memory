from flask import Flask, request, jsonify

import csv
import os
from datetime import datetime
import cv2
import numpy as np
import pickle
from PIL import Image, ImageOps
import io
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from flask import send_file

app = Flask(__name__)
present_students = set()
print("Loading FaceNet...")
embedder = FaceNet()

print("Loading MTCNN...")
detector = MTCNN()

print("Loading Database...")
with open("student_embeddings.pkl", "rb") as f:
    database = pickle.load(f)

print("Students Loaded:", len(database))

if "HIMESH RINCHHODIYA" in database:
    
    print(
        "HIMESH COUNT =",
        len(
            database["HIMESH RINCHHODIYA"]["Embeddings"]
        )
    )


def recognize_face_crop(face_crop):

    face_crop = cv2.resize(
        face_crop,
        (160, 160)
    )

    face_crop = np.expand_dims(
        face_crop,
        axis=0
    )
    
    print("Embedding Generated")

    embedding = embedder.embeddings(
        face_crop
    )[0]

    best_match = "Unknown"
    best_score = -1

    for student, info in database.items():

        embedding_list = info["Embeddings"]

        for stored_embedding in embedding_list:

            score = cosine_similarity(
                [embedding],
                [stored_embedding]
            )[0][0]

            if score > best_score:

                best_score = score
                best_match = student

    return best_match, float(best_score)



def mark_attendance(student_name):
    
    file_name = "attendance.csv"

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    existing = set()

    if os.path.exists(file_name):

        with open(file_name,"r") as file:

            reader = csv.reader(file)

            next(reader,None)

            for row in reader:

                if len(row) >= 2:

                    existing.add(
                        (row[0],row[1])
                    )

    if (student_name,today) in existing:

        return

    file_exists = os.path.isfile(file_name)

    with open(
        file_name,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Name",
                "Date",
                "Time"
            ])

        writer.writerow([
            student_name,
            today,
            current_time
        ])
        
@app.route("/download_attendance")
def download_attendance():

    return send_file(
    "attendance.csv",
    as_attachment=True,
    download_name="attendance.csv",
    mimetype="text/csv"
    )


@app.route("/recognize", methods=["POST"])
def recognize():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]
    
    source = request.form.get(
    "source",
    "camera"
    )

    print(
    "SOURCE:",
    source
    )

    pil_image = Image.open(
    io.BytesIO(file.read())
    )

    pil_image = ImageOps.exif_transpose(
    pil_image
    )
    
    
    pil_image.save("pil_fixed.jpg")
    
    print("PIL SIZE:", pil_image.size)

    

    image = cv2.cvtColor(
    np.array(pil_image),
    cv2.COLOR_RGB2BGR
    )
    
    cv2.imwrite("fixed_orientation.jpg", image)

    
    
    cv2.imwrite(
    "debug_upload.jpg",
    image
    )

    print(
    "Uploaded Image Shape:",
    image.shape
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    
    h, w = rgb.shape[:2]

    max_size = 1280

    scale = 1.0

    if max(h, w) > max_size:

     scale = max_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    rgb = cv2.resize(
    rgb,
    (new_w, new_h)
    )

    print(
    "Resized Shape:",
    rgb.shape
    )

    
    
    cv2.imwrite("after_resize.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite("original_upload.jpg", image)
    print("Final Shape:", rgb.shape)

    if source == "camera":
    
     print("CAMERA MODE")

    faces = detector.detect_faces(rgb)

    print(
        "Faces Found:",
        len(faces)
    )

    faces = [

        face

        for face in faces

        if face.get(
            "confidence",
            1.0
        ) >= 0.90
    ]

    if len(faces) > 1:

        faces = sorted(

            faces,

            key=lambda f:
            f["box"][2] *
            f["box"][3],

            reverse=True

        )

        faces = [faces[0]]

    else:

     print("GALLERY MODE")

    best_faces = []
    best_image = rgb

    rotations = [

        rgb,

        cv2.rotate(
            rgb,
            cv2.ROTATE_90_CLOCKWISE
        ),

        cv2.rotate(
            rgb,
            cv2.ROTATE_180
        ),

        cv2.rotate(
            rgb,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    ]

    for img in rotations:

        detected = detector.detect_faces(img)

        if len(detected) > len(best_faces):

            best_faces = detected
            best_image = img

    faces = best_faces
    rgb = best_image

    print("Faces Detected:", len(faces))

    

       

    recognized_students = []

    for face_data in faces:

        try:

            x, y, w, h = face_data["box"]

            x = max(0, x)
            y = max(0, y)

            face_crop = rgb[
                y:y+h,
                x:x+w
            ]
            
            print("Crop Shape:", face_crop.shape)

            if face_crop.size == 0:
                continue

            name, score = recognize_face_crop(
                face_crop
            )
            
            
            print("Prediction:",name,"Score:",score)

            if score >= 0.75:

                mark_attendance(name)
                present_students.add(name)
                recognized_students.append({

                "name": name,

                "score": round(score,4)

                })

        except Exception as e:

            print(e)

    return jsonify({

        "total_faces": len(faces),

        "recognized_students":
        recognized_students

    })
    
    
@app.route("/export_attendance")
def export_attendance():

    import csv
    from datetime import datetime

    file_name = "attendance_report.csv"

    total_students = len(database)

    present_count = len(present_students)

    absent_count = total_students - present_count

    with open(
        file_name,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            datetime.now().strftime("%d-%m-%Y")
        ])

        writer.writerow([])

        writer.writerow([
            "Total Students",
            total_students
        ])

        writer.writerow([
            "Present",
            present_count
        ])

        writer.writerow([
            "Absent",
            absent_count
        ])

        writer.writerow([])

        writer.writerow([
            "Enrollment",
            "Name",
            "Branch",
            "Semester",
            "Section",
            "Status"
        ])

        for name, info in database.items():

            status = "Present"

            if name not in present_students:

                status = "Absent"

            writer.writerow([

                info.get(
                    "Enrollment_No",
                    ""
                ),

                name,

                info.get(
                    "Branch",
                    ""
                ),

                info.get(
                    "Semester",
                    ""
                ),

                info.get(
                    "Section",
                    ""
                ),

                status

            ])

    return jsonify({
        "status":"success"
    })    
    
    
    
@app.route("/add_student", methods=["POST"])
def add_student():

    try:

        name = request.form["name"].strip().upper()
        print("Received Name:", name)
        enrollment = request.form["enrollment"]
        branch = request.form["branch"]
        semester = request.form["semester"]
        section = request.form["section"]

        file = request.files["image"]

        image = Image.open(io.BytesIO(file.read()))

        image = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        faces = detector.detect_faces(rgb)
        
        print("Student:", name)
        print("Faces Found:", len(faces))

        if len(faces) == 0:

            return jsonify({
                "status":"error",
                "message":"No face detected"
            })

        x,y,w,h = faces[0]["box"]

        x = max(0,x)
        y = max(0,y)

        face_crop = rgb[y:y+h,x:x+w]
        
        print("Crop Shape:", face_crop.shape)
        
        if face_crop.size == 0:
    
         print("EMPTY FACE CROP")

         return jsonify({"status":"error","message":"Empty Crop"})

        face_crop = cv2.resize(
            face_crop,
            (160,160)
        )
        
        cv2.imwrite(
        f"debug_{name}.jpg",
       cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
        )
        
        print("Generating Embedding...")

        embedding = embedder.embeddings(
            np.expand_dims(
                face_crop,
                axis=0
            )
        )[0]

        database[name] = {

    "Enrollment_No": enrollment,
    "Branch": branch,
    "Semester": semester,
    "Section": section,
    "Embeddings": [embedding]

     }

        with open(
            "student_embeddings.pkl",
            "wb"
        ) as f:

            pickle.dump(
                database,
                f
            )

        return jsonify({

            "status":"success",
            "student":name

        })
        
        

    except Exception as e:

        return jsonify({

            "status":"error",
            "message":str(e)

        })
        
        
@app.route("/add_face_sample", methods=["POST"])
def add_face_sample():

    try:

        name = request.form["name"].strip().upper()

        print("\n========================")
        print("ADD FACE SAMPLE CALLED")
        print("NAME:", name)
        print("========================")

        if name not in database:

            print("STUDENT NOT FOUND")

            return jsonify({
                "status":"error",
                "message":"Student not found"
            })

        file = request.files["image"]

        image = Image.open(
            io.BytesIO(file.read())
        )

        image = cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        print(
            "UPLOADED IMAGE SHAPE:",
            rgb.shape
        )

        # ----------------------------------
        # IMPORTANT CHANGE
        # Android already sends cropped face
        # Don't run MTCNN again
        # ----------------------------------

        print("USING ANDROID FACE CROP")

        face_crop = rgb

        if face_crop.size == 0:

            return jsonify({
                "status":"error",
                "message":"Empty Face Crop"
            })

        face_crop = cv2.resize(
            face_crop,
            (160,160)
        )

        print("START EMBEDDING")

        embedding = embedder.embeddings(
            np.expand_dims(
                face_crop,
                axis=0
            )
        )[0]

        print("EMBEDDING GENERATED")

        database[name]["Embeddings"].append(
            embedding.tolist()
        )

        print(
            "NEW COUNT =",
            len(
                database[name]["Embeddings"]
            )
        )

        with open(
            "student_embeddings.pkl",
            "wb"
        ) as f:

            pickle.dump(
                database,
                f
            )

        return jsonify({

            "status":"success",

            "total_samples":
            len(
                database[name]["Embeddings"]
            )

        })

    except Exception as e:

        print("\nADD FACE SAMPLE ERROR")
        print(e)

        import traceback
        traceback.print_exc()

        return jsonify({

            "status":"error",
            "message":str(e)

        })
        
@app.route("/delete_student", methods=["POST"])
def delete_student():

    try:

        name = request.form["name"].strip().upper()

        if name not in database:

            return jsonify({

                "status":"error",
                "message":"Student not found"

            })

        del database[name]

        with open(
            "student_embeddings.pkl",
            "wb"
        ) as f:

            pickle.dump(
                database,
                f
            )

        return jsonify({

            "status":"success"

        })

    except Exception as e:

        return jsonify({

            "status":"error",
            "message":str(e)

        })
 
 
      
        
@app.route("/students", methods=["GET"])
def get_students():

    students = []

    for name, info in database.items():

      students.append(

    f"{name}\n"
    f"{info.get('Enrollment_No','')}\n"
    f"{info.get('Branch','')}\n"
    f"Semester: {info.get('Semester','')}\n"
    f"Section: {info.get('Section','')}"
       )

    return jsonify(students)
        
              
@app.route("/update_student", methods=["POST"])
def update_student():

    name = request.form["name"]
    
    print("UPDATE REQUEST")
    print(request.form)

    print("UPDATE REQUEST")
    print("Name:", name)

    if name not in database:

        print("NOT FOUND IN DATABASE")

        return jsonify({
            "status":"error",
            "message":"Student not found"
        })

    print("BEFORE UPDATE")
    print(database[name])

    database[name]["Enrollment_No"] = request.form["enrollment"]
    database[name]["Branch"] = request.form["branch"]
    database[name]["Semester"] = request.form["semester"]
    database[name]["Section"] = request.form["section"]

    print("AFTER UPDATE")
    print(database[name])

    with open(
        "student_embeddings.pkl",
        "wb"
    ) as f:

        pickle.dump(
            database,
            f
        )

    return jsonify({
        "status":"success"
    })
    

@app.route("/attendance", methods=["GET"])
def attendance():

    print("ATTENDANCE API CALLED")

    records = []

    try:
        with open("attendance.csv", "r") as file:

            lines = file.readlines()

            for line in lines[1:]:

                records.append(line.strip())

    except Exception as e:

        print(e)

    return jsonify(records)  



@app.route("/student_samples/<name>")
def student_samples(name):

    name = name.upper()

    if name not in database:
        return jsonify({"count":0})

    return jsonify({
        "count":
        len(
            database[name]["Embeddings"]
        )
    })   
    
@app.route("/all_students")
def all_students():

    return jsonify(
        list(database.keys())
    )                 


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )