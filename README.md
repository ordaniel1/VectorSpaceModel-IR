# VectorSpaceModel-IR
This repository contains a Python implementation of an Information Retrieval system based on the Vector Space Model (VSM). The system creates an inverted index, supports querying with TF-IDF and BM25 as ranking functions, and ranks relevant documents.

## Usage

### Create Inverted Index

To create the inverted index, run the following command:

```bash
python vsm_ir.py create_index [corpus_directory]
```
### Query

To perform a query using either TF-IDF or BM25, use the following command:

```bash
python vsm_ir.py query [ranking] [index_path] “<question>”
```
-[ranking]: Choose the ranking function, either bm25 or tfidf.
-[index_path]: Provide the path to the inverted index.
-<question>: Enter an English question in quotes.

## Dependencies
-Python 3.x
-NLTK library



