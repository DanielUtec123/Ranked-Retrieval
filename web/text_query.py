# ------------------PROJECT2-PART-1-TF-IDF-RANKED-BASED-RETRIEVAL-DANIEL-ROJAS-FELIX-SOLANO-----------------------------
import nltk
import time
from math import log, sqrt
# default dict se usa para evitar crear una nueva entrada vacia de diccionario cada vez que haya una nueva palabra
from collections import defaultdict
import json
# from numpy import unicode

total_doc_count = 500
inv_index = defaultdict(list)  # Retorna una lista vacia cada vez que se accede a un elemento inexistente
all_doc_vectors = []  # Cada elemento de la lista es un diccionario, existira un vector para cada documento
doc_freq = {}


# Agrega tf dicts (como elementos de una lista) a la lista llamada all_doc_vectors
def read_all_docs():
    with open('test1.json') as f:
        d = json.load(f)
        for i in range(total_doc_count):
            s = d[i]
            tx = s['text']
            # print(tx)
            token_lst = stem_and_tokenize(tx)

            v = create_vector(token_lst)
            all_doc_vectors.append(v)
    # for doc_id in range(total_doc_count - 1):
    #     doc_text = doc_string(doc_id)
    #     token_lst = stem_and_tokenize(doc_text)
    #     v = create_vector(token_lst)
    #     all_doc_vectors.append(v)


# Crea un diccionario tf de la query ingresada
def input_vector(theQuery):
    v = {}
    for word in theQuery:
        if word in v:
            v[word] += 1.0  # Float ya que se convertiran en TF-IDF mas tarde
        else:
            v[word] = 1.0
    return v


# Generates inverted index for all documents.
def inv_index_all_docs():
    count = 0
    for doc_vector in all_doc_vectors:
        for word in doc_vector:
            inv_index[word].append(count)  # Aqui defaul dict muestra su valor.
        count += 1


# Cambia todas los vectores TF a vectores TF-IDF
def tf_idf_vectorized():
    length = 0
    for doc_vector in all_doc_vectors:
        for word in doc_vector:
            frequency = doc_vector[word]
            score = tf_idf_score(word, frequency)
            doc_vector[word] = score
            length += score ** 2
        length = sqrt(length)
        for word in doc_vector:
            doc_vector[word] /= length


# Calculates the TF-IDF vector for the query in specific.
def tf_idf_query(query_vec):
    length = 0.0
    for word in query_vec:
        frequency = query_vec[word]
        if word in doc_freq:
            query_vec[word] = tf_idf_score(word, frequency)
        else:
            query_vec[word] = log(1 + frequency) * log(total_doc_count)
        length += query_vec[word] ** 2
    length = sqrt(length)
    if length != 0:
        for word in query_vec:
            query_vec[word] /= length


# Calcula el TF-IDF score, dado un TF y un DF
def tf_idf_score(word, frequency):
    return log(1 + frequency) * log(total_doc_count / doc_freq[word])


# Calcula el producto punto dado dos vectores
def dot_product(vector_a, vector_b):
    if len(vector_a) > len(vector_b):  # Swapping para asegurarse que la parte izquiera del dict siempre es mas pequena
        temp = vector_a
        vector_a = vector_b
        vector_b = temp
    key_list_a = vector_a.keys()
    key_list_b = vector_b.keys()
    res = 0
    for key in key_list_a:
        if key in key_list_b:
            res += vector_a[key] * vector_b[key]
    return res


# Retorna una lista de tokens despues del stemming
def stem_and_tokenize(doc_text):
    tkn_list = nltk.word_tokenize(doc_text)
    ps = nltk.stem.snowball.SnowballStemmer("spanish")
    my_result = []
    for word in tkn_list:
        my_result.append(ps.stem(word))
    return my_result


# Creates token-frequency vector from input string.
def create_vector(the_token_list):
    v = {}
    global doc_freq
    for token in the_token_list:
        if token in v:
            v[token] += 1
        else:
            v[token] = 1
            if token in doc_freq:
                doc_freq[token] += 1
            else:
                doc_freq[token] = 1
    return v


# Lee data de conjunto de archivos en el Conjunto de Datos presente en el mismo directorio que este script
def doc_string(doc_id):
    file_text = str(open("Files/"+str(doc_id)).read())
    return file_text


# Retorna una lista de IDs de documentos ordenada y basada en la similitud de coseno
def query_result(q_vector):
    answer = []
    user_idList = list()
    list_text = []
    with open('test1.json') as f:
        d = json.load(f)
        for i in range(total_doc_count):
            s = d[i]
            tx = s['id']
            text = s['text']
            list_text.append(text)
            # clDoc = ' '.join(tx)
            user_idList.append(tx)
    # print(user_idList)
    for doc_id in range(total_doc_count - 1):
        dp = dot_product(q_vector, all_doc_vectors[doc_id])
        answer.append((user_idList[doc_id], dp,list_text[doc_id]))
    print(answer)
    answer = sorted(answer, key=lambda x: x[1], reverse=True)
    print(answer)
    # for doc_id in range(total_doc_count - 1):
    #     dp = dot_product(q_vector, all_doc_vectors[doc_id])
    #     answer.append((doc_id, dp))
    # answer = sorted(answer, key=lambda x: x[1], reverse=True)
    return answer


# -----------------------------------Desde-aqui-empieza-la-ejecucion---------------------------------------------------
# EMPIEZA EL PREPROCESAMIENTO
start_index_time = time.time()
read_all_docs()
inv_index_all_docs()
tf_idf_vectorized()
# TERMINA EL PREPROCESAMIENTO


#
#
#
#
#
#
# print("Indexing took " + str(time.time() - start_index_time) + " seconds.")
# start_search_time = time.time()
# print("This query took " + str(time.time()-start_search_time) + " seconds.")
