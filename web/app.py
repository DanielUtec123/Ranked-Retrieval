from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template,redirect, request, session, Response,jsonify
import json
#import tf_idf
import text_query

from datetime import datetime


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rankedRetrieval')
def rankedRetrieval():
    return render_template('ranked.html')


@app.route('/imageRecovering', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return image_query.detect_faces_in_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return render_template('imageRecover.html')






@app.route('/do_image_recovering/<query>', methods=['POST','GET'])



#retorna un json con los resultados
@app.route('/do_ranked_search/<query>', methods=['POST','GET'])
def get_ranked_results_by_query(query):
    if len(query) == 0:
        return
    results = []

    token_list = text_query.stem_and_tokenize(query)
    query_vector = text_query.input_vector(token_list)
    text_query.tf_idf_query(query_vector)
    result = text_query.query_result(query_vector)
    f_result = result[:10]
    # print(f_result)
    for element in f_result:
        record = {}
        record['name'] = str(element[0])
        record['weight'] = str(element[1])
        record['text'] = str(element[2])
        results.append(record)
        print("The DocID " + str(element[0]) + " matches, with weight " + str(element[1]))

    print(query)
    print(results)
    return jsonify(results)



if __name__ == '__main__':
    app.run(debug=True)


''' results = []
    if len(query) > 0:

        token_list = tf_idf.stem_and_tokenize(query)
        query_vector = tf_idf.input_vector(token_list)
        tf_idf.tf_idf_query(query_vector)
        result = tf_idf.query_result(query_vector)
        f_result = result[:10]
        for element in f_result:
            record = {}
            record['name'] = str(element[0])
            record['weight'] =str(element[1])
            results.append(record)
            print("The DocID " + str(element[0]) + " matches, with weight " + str(element[1]))


    print(query)
    print(results)
    return jsonify(results)'''

