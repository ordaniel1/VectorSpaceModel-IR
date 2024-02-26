import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import os
import sys
import math
import json
import string
from nltk.stem import PorterStemmer

try:
    sw_nltk = stopwords.words('english')
except:
    nltk.download('stopwords')
    sw_nltk = stopwords.words('english')

stemmer = PorterStemmer()
exclude = set(string.punctuation) # !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
N = 1239  # total number of documents in corpus


def create_inverted_index(corpus_directory):
    """ The function receives a directory of a corpus and creates an inverted index."""

    H = {}  # inverted index
    D_lengths = {}  # length of vectors (not length in words)

    # loop through all xml files
    sum_lengths = 0 # number of words in all records
    for filename in os.listdir(corpus_directory):
        if (not filename.endswith('.xml') or filename == "cfquery.xml"):
            continue

        fullname = os.path.join(corpus_directory, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()

        records = root.findall('RECORD')  # get all records (= files)
        # loop through all records in each file
        for record in records:
            (V, total_words, max_rep) = create_vector(record) # create vector of the record
            record_number = record.find('RECORDNUM').text  # pull record number
            D_lengths[record_number] = 0.0  # initialize document's (vector) length
            sum_lengths += total_words  # total words in all the records together - for later average length calculation

            # loop through all terms in each record
            for term in V:
                if term not in H:  # term is not in our dictionary yet
                    H[term] = []  # initialize empty list
                # list of tuples - each of the form (<record_num>, <number of appearances in the specific file>, max_rep for the record, total_words in the record)
                H[term].append((record_number, V[term], max_rep, total_words))

    # compute IDF for all Tokens in H and save it in the end of T's occList (T=term)
    for term in H.keys():
        idf = math.log2(N / len(H[term]))
        H[term].append(idf)

    # compute length of all document vectors (according to tf-idf weights):
    for term in H.keys():
        idf = H[term][-1]

        for i in range(len(H[term]) - 1):  # for each D (for each document except of the last item in T's occList)
            record_number = H[term][i][0]  # record number of D
            tf = H[term][i][1] / H[term][i][2]  # tf(T,D) number of appearences of term i in record j / max repetition of some word in record j

            D_lengths[record_number] += (tf * idf) ** 2  # increment D's length

    for record_number in D_lengths.keys():
        D_lengths[record_number] = math.sqrt(D_lengths[record_number])

    H["D_lengths"] = D_lengths
    H["avg_length_in_words"] = sum_lengths / N

    return H


def create_vector(record):
    """ # the method gets a record and returns a 3-tuple:
    1. dictionary of the form - key:term , value:number of appearances (after moving out stopwords and stemming).
    2. int - total words in the record.
    3. int - number of appearances of the most frequented word in the record."""

    V = {}

    # extract the relevant parts of the record
    text = record.find('TITLE').text
    if(record.find('EXTRACT') != None):
        text +=" "
        text += record.find('EXTRACT').text
    if (record.find('ABSTRACT') != None):
        text +=" "
        text += record.find('ABSTRACT').text
    updated_text = modify_text(text) # take of special characters

    # filter the words in the text
    text_array = [word.lower() for word in updated_text.split() if word.lower() not in sw_nltk] # take of stop words and turn into lower case
    text_array = [stemmer.stem(word) for word in text_array] # stem the words
    total_words = len(text_array)

    # count number of appearances for each term & update the dictionary
    for word in text_array:
        if word in V:  # word is already in the dictionary
            V[word] += 1
        else:  # insert key to dict
            V[word] = 1

    max_rep = max(V.values()) # max number of appearances for a term in this Record

    return (V, total_words, max_rep)


def modify_text(text):
    """ The function receives a text (string) and returns a text without special characters."""
    updated_text = ""
    for i in text:
        if i in exclude or i in digits:
            updated_text += " "  # replace all special characters with space (= " ")
        else:
            updated_text += i

    return updated_text


def create_query_vector(q):
    """ The function receives a query (string) and returns a hashmap: key: term, value: f(Q,t)"""
    words = [word.lower() for word in q.split() if word.lower() not in sw_nltk] #tokenize and delete stopwords
    words=[stemmer.stem(word) for word in words] #stemming
    Q = {} #create the hashmap
    for word in words:
        if word in Q:
            Q[word] += 1
        else:
            Q[word] = 1
    return Q


def retrieval_tfidf(Q, H):
    """ The function receives a query (vector) and an inverted index and returns a list of the most relevant records"""
    R = {}  # store relevant documents with scores
    S = 0  # sum of squares of Q's weights

    max_rep = max(Q.values()) # max appearances of a term in the filtered query
    for term in Q.keys():
        if term not in H.keys(): # for a query term which is not in our corpus
            continue

        idf = H[term][-1]  # IDF(T)
        tf_q = Q[term] / max_rep

        w = tf_q * idf  # weight of token T in query Q
        S += w**2

        for i in range(len(H[term])-1):  # for every D (doc) that T (term) appears in
            record_number = H[term][i][0]  # record number of D
            tf_d = H[term][i][1] / H[term][i][2]  # tf(D,T)

            if record_number not in R:
                R[record_number] = 0.0

            R[record_number] += (idf * tf_d) * w  # w_ij * w_iq

    L = math.sqrt(S)  # length(Q)

    for record_number in R.keys(): #calculate final scores (cosine-similarity)
        Y = H["D_lengths"][record_number] #length of document (vector) according to tf-idf weights
        R[record_number] = R[record_number] / (Y * L)

    # keep the records with higher degree than 0.075 and return list of their numbers (in decreasing order of their scores)
    newR={ key: R[key] for key in R.keys() if R[key] >= 0.075}
    return sorted(newR, key= lambda key: -newR[key])


def retrieval_bm25(Q, H):
    """ The function receives a query (vector) and an inverted index, and returns a list of the most relevant records"""

    R = {} # dictionary to hold the degrees for the different records
    avg = H["avg_length_in_words"] #average length (in words) of a document in corpus
    b = 0.75  # b is a bm25 parameter (0.75 recommended choice according to the slides)
    k = 1.5  # k is a bm25 parameter (in the range [1.2,2.0])

    for term in Q.keys(): # for each tern in the query

        if term not in H.keys(): # if the term is not in the corpus
            continue
        idf = math.log((N - (len(H[term]) - 1) + 0.5) / ((len(H[term]) - 1) + 0.5) + 1) #calculate idf(term) according to bm25 formula
        for i in range(len(H[term])-1): # for every D (doc) that T (term) appears in
            record_number = H[term][i][0]  # record number of D
            f = H[term][i][1]  # count of T (term) in D
            D_size = H[term][i][3] #total words in D

            if record_number not in R:
                R[record_number] = 0.0
            R[record_number] += idf * ((f * (k+1)) / (f + k * (1 - b + (b * (D_size / avg))))) #incerment BM25(D,Q) score

    # keep the records with higher degree than 7 and return a list of their numbers (in decreasing order of their scores)
    newR={ key: R[key] for key in R.keys() if R[key] >= 7}
    return sorted(newR, key = lambda key: -newR[key])


def main():

    args = sys.argv[1:]  # get user's arguments
    if(len(args) == 2 and args[0] == "create_index"): #create an inverted index
        corpus_directory = args[1]
        H = create_inverted_index(corpus_directory)

        # save the following file in same directory as the python file
        # vsm_inverted_index.json
        file_name = 'vsm_inverted_index.json'
        with open(file_name, 'w') as file_object:
            json.dump(H, file_object)

    elif(len(args) == 4 and args[0] == "query" and (args[1] == "tfidf" or args[1] == "bm25")): #IR case
        ranking = args[1]
        index_path = args[2]
        question = args[3]

        question = modify_text(question)
        Q = create_query_vector(question)

        # load the inverted index
        with open(index_path, 'r') as file_object:
            H = json.load(file_object)

        # keep the process with tf-idf || BM25
        if (ranking == "tfidf"):  # tf-idf
            docs = retrieval_tfidf(Q, H) #list of relevant documents
            f=open('ranked_query_docs.txt','w')
            for i in range(len(docs)):
                doc=str(int(docs[i]))
                print(doc, file=f)
            f.close()

        elif (ranking == "bm25"):  # BM25

            docs = retrieval_bm25(Q,H) #list of relevant documents
            f = open('ranked_query_docs.txt', 'w')
            for i in range(len(docs)):
                doc=str(int(docs[i]))
                print(doc, file=f)
            f.close()



if __name__ == "__main__":
   main()
