import face_recognition
from flask import Flask, jsonify, request, redirect
import math
import os
import operator

# You can change this to any folder on your system


def euclidean_distance(vec1,vec2):
    s = 0
    for i in range(len(vec1)):
        s += math.pow(vec1[i] - vec2[i],2)
    s = math.sqrt(s)
    return s

def detect_faces_in_image(file_stream):
     # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)

     distance_list = {}

     picture = face_recognition.load_image_file(file_stream)
     known_face_encoding = face_recognition.face_encodings(picture)[0]


     with open("ImageEncoding.txt", "r") as file:
         for linea in file:
             linea = linea.strip("\n").split(" ")
             encoding = linea[0:128]
             encoding = [float(i) for i in encoding]
             distance_list[linea[129]] = [euclidean_distance(known_face_encoding, encoding), linea[128]]
     resultado = sorted(distance_list.items(), key=operator.itemgetter(1))

     results = []
     for i in range(10):
         record ={}
         record['name'] = resultado[i][0]
         img_p = resultado[i][1][1][49:]
         record['img_path'] = img_p
         record['score'] = resultado[i][1][0]
         results.append(record)
    # Return the result as json
     print(results)

     html = ""
     for i in range(len(results)):
        html = html + '<img src="/static/'+results[i]['img_path'] +'"width="400"height="341">'
        html = html + "<h1>"+ str(results[i]['name'])+"</h1>"
        html = html + "<h1>" + str(results[i]['score']) + "</h1>"
     return html

#recorremos todas las imagenes

'''
for dir in os.listdir(os.getcwd()+"/lfw"):
    imgs =os.listdir(os.getcwd()+"/lfw/"+ dir)
    img =  imgs[0]
    print(os.getcwd()+"/lfw/"+ dir + "/" + img)
    img_path =os.getcwd()+"/lfw/"+ dir + "/" + img
    image = face_recognition.load_image_file(img_path)
    if len(face_recognition.face_encodings(image))>0:
        unknown_face_encodings = face_recognition.face_encodings(image)[0]
        distance_list.append(euclidean_distance(known_face_encoding,unknown_face_encodings))
        print(unknown_face_encodings)
        for x in unknown_face_encodings:
            file.write(str(x)+" ")
        file.write(img_path+ " ")
        file.write(dir+ "\n")
file.close()
'''

#print(detect_faces_in_image("fotos_test/foto1.jpg"))


