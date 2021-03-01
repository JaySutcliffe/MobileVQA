"""
Adapted from https://github.com/chingyaoc/VQA-tensorflow
"""
from random import shuffle, seed
import sys
import numpy as np
import argparse
import h5py
from nltk.tokenize import word_tokenize
import json
import re


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if
            i != '' and i != ' ' and i != '\n']


def prepro_question(imgs, params):
    # preprocess all the question
    print('example processed tokens:')
    for i, img in enumerate(imgs):
        s = img['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print(txt)
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" % (i, len(imgs), i * 100.0 / len(imgs)))
            sys.stdout.flush()
    return imgs


def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets now produce the final annotation
    # additional special UNK token we will use below to map infrequent words to
    print('inserting the special UNK token')
    vocab.append('UNK')

    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab


def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w, len(wtoi) + 1) != (len(wtoi) + 1) else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs


def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans']
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top answer and their counts:')
    print('\n'.join(map(str, cw[:20])))

    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]


def encode_question(imgs, params, wtoi):
    max_length = params['max_length']
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    for i, img in enumerate(imgs):
        question_id[question_counter] = img['ques_id']
        label_length[question_counter] = min(max_length,
                                             len(img['final_question']))  # record the length of this sequence
        question_counter += 1
        for k, w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i, k] = wtoi[w]

    return label_arrays, label_length, question_id


def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ix = atoi.get(img['ans'])
        if ix is None:
            ans_arrays[i] = -1  # default value to -1 if not present
        else:
            ans_arrays[i] = ix

    return ans_arrays


def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if atoi.get(img['ans'], len(atoi) + 1) != len(atoi) + 1:
            new_imgs.append(img)

    print('question number reduce from %d to %d ' % (len(imgs), len(new_imgs)))
    return new_imgs


def get_unique_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['img_path']] = count_img.get(img['img_path'], 0) + 1

    unique_img = [w for w, n in count_img.items()]
    imgtoi = {w: i + 1 for i, w in enumerate(unique_img)}  # add one for torch, since torch start from 1.

    for i, img in enumerate(imgs):
        img_pos[i] = imgtoi.get(img['img_path'])

    return unique_img, img_pos


def answers_to_strings(data, atoi):
    strings = []
    max_length = 0
    for i in range(0, len(data)):
        answers = data[i]['answers']
        string = ""
        for answer, count in answers.items():
            ix = atoi.get(answer)
            if ix is not None:
                string = string + str(ix) + "," + str(count) + ";"

        strings.append(np.string_(string[:-1]))
        if len(string) - 1 > max_length:
            max_length = len(string) - 1
    return strings, max_length


def main(params):
    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w: i + 1 for i, w in enumerate(top_ans)}
    itoa = {i + 1: w for i, w in enumerate(top_ans)}

    # filter question, which isn't in the top answers
    if params['filter']:
        imgs_train = filter_question(imgs_train, atoi)

    seed(123)  # make reproducible
    shuffle(imgs_train)  # shuffle the order

    # tokenization and preprocessing training question
    imgs_train = prepro_question(imgs_train, params)
    # tokenization and preprocessing testing question
    imgs_test = prepro_question(imgs_test, params)

    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, params, wtoi)

    imgs_test = apply_vocab_question(imgs_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(imgs_test, params, wtoi)

    # get the unique image for train and test
    unique_img_train, img_pos_train = get_unique_img(imgs_train)
    unique_img_test, img_pos_test = get_unique_img(imgs_test)

    # get the answer encoding.
    A_train = encode_answer(imgs_train, atoi)
    A_test = encode_answer(imgs_test, atoi)

    A_more_train, max_length_train = answers_to_strings(imgs_train, atoi)
    A_more_test, max_length_test = answers_to_strings(imgs_test, atoi)

    # create output h5 file for training set.
    N = len(imgs_train)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("ques_train", dtype='uint32', data=ques_train)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train)
    f.create_dataset("question_id_train", dtype='uint32', data=question_id_train)
    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    f.create_dataset("ans_train", dtype='uint32', data=A_train)
    f.create_dataset("ans_more_train", dtype='S'+str(max_length_train), data=A_more_train)

    f.create_dataset("ques_test", dtype='uint32', data=ques_test)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test)
    f.create_dataset("question_id_test", dtype='uint32', data=question_id_test)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)
    f.create_dataset("ans_test", dtype='uint32', data=A_test)
    f.create_dataset("ans_more_test", dtype='S'+str(max_length_test), data=A_more_test)

    f.close()
    print('wrote ', params['output_h5'])

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    out['unique_img_test'] = unique_img_test
    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='data/vqa_raw_train.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='data/vqa_raw_test.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default='3000', type=int,
                        help='number of top answers for the final classifications.')
    parser.add_argument('--filter', default=False, type=bool,
                        help='whether to filter out questions not in the top answers')

    parser.add_argument('--output_json', default='data/data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='data/data_prepro.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=26, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int,
                        help='number of test images (to withold until very very end)')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')

    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('Parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
