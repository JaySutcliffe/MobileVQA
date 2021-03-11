import argparse
import numpy as np
import json


def main(input_json, glove_location, output_npy):
    """
    Takes in a in the input file, GloVe location and generates an
    output npy file containing the embeddings.

    Parameters:
        input_json (str): input json file generated from preprocess_questions.npy
        glove_location (str): the location of the GloVe word embeddings
        output_npy (str): the location and file name to store the embedding matrix

    Adapted from code taken from...
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    on 17/02/2021
    """
    embeddings_index = {}

    f = open(glove_location, 'r', encoding="utf-8")
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    with open(input_json) as data_file:
        data = json.load(data_file)
        embedding_matrix = np.zeros((len(data['ix_to_word']) + 5, 300))  # +5 for empty etc

    for i, word in data['ix_to_word'].items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # if the word is not present in the matrix it will be all 0s
            embedding_matrix[int(i)] = embedding_vector

    with open(output_npy, "wb") as f:
        np.save(f, embedding_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='data/data_prepro.json',
                        help='file containing index to word table')
    parser.add_argument('--glove_location', default='data/glove.840B.300d.txt',
                        help='place to put output npy embedding file')
    parser.add_argument('--output_npy', default='data/glove.npy',
                        help='place to put output npy embedding file')

    args = parser.parse_args()
    params = vars(args)
    main(params['input_json'], params['glove_location'], params['output_npy'])