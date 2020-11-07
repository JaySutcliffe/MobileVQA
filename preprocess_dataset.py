"""
Code adapted from https://github.com/chingyaoc/VQA-tensorflow/
Contributor: Jiasen Lu
"""

import json
import argparse

def main(params):
    '''
    Put the VQA data into single json file, where [[Question_id, Image_id, Question, Answer] ... ]
    '''

    train = []
    test = []
    imdir = params['dir'] + '/{0}/COCO_{0}_{1:012d}.jpg'

    train_annotations_file = params['dir'] + '/v2_mscoco_train2014_annotations.json'
    val_annotations_file = params['dir'] + '/v2_mscoco_val2014_annotations.json'
    train_questions_file = params['dir'] + '/v2_OpenEnded_mscoco_train2014_questions.json'
    val_questions_file = params['dir'] + '/v2_OpenEnded_mscoco_val2014_questions.json'
    test_questions_file = params['dir'] + '/v2_Questions_Test_mscoco/v2_OpenEnded_mscoco_test2015_questions'


    if params['split'] == 1:

        print('Loading annotations and questions...')
        train_anno = json.load(open(train_annotations_file, 'r'))
        val_anno = json.load(open(val_annotations_file, 'r'))

        train_ques = json.load(open(train_questions_file, 'r'))
        val_ques = json.load(open(val_questions_file, 'r'))

        subtype = 'train2014'
        for i in range(len(train_anno['annotations'])):
            ans = train_anno['annotations'][i]['multiple_choice_answer']
            question_id = train_anno['annotations'][i]['question_id']
            image_path = imdir.format(subtype, train_anno['annotations'][i]['image_id'])

            question = train_ques['questions'][i]['question']

            train.append(
                {'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})

        subtype = 'val2014'
        for i in range(len(val_anno['annotations'])):
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            question_id = val_anno['annotations'][i]['question_id']
            image_path = imdir.format(subtype, val_anno['annotations'][i]['image_id'])

            question = val_ques['questions'][i]['question']

            test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})
    else:
        print('Loading annotations and questions...')
        train_anno = json.load(open(train_annotations_file, 'r'))
        val_anno = json.load(open(val_annotations_file, 'r'))

        train_ques = json.load(open(train_questions_file, 'r'))
        val_ques = json.load(open(val_questions_file, 'r'))
        test_ques = json.load(open(test_questions_file, 'r'))

        subtype = 'train2014'
        for i in range(len(train_anno['annotations'])):
            ans = train_anno['annotations'][i]['multiple_choice_answer']
            question_id = train_anno['annotations'][i]['question_id']
            image_path = imdir.format(subtype, train_anno['annotations'][i]['image_id'])

            question = train_ques['questions'][i]['question']

            train.append(
                {'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})

        subtype = 'val2014'
        for i in range(len(val_anno['annotations'])):
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            question_id = val_anno['annotations'][i]['question_id']
            image_path = imdir.format(subtype, val_anno['annotations'][i]['image_id'])

            question = val_ques['questions'][i]['question']

            train.append(
                {'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})

        subtype = 'test2015'
        for i in range(len(test_ques['questions'])):
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            question_id = test_ques['questions'][i]['question_id']
            image_path = imdir.format(subtype, test_ques['questions'][i]['image_id'])

            question = test_ques['questions'][i]['question']

            test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})

    print('Training sample %d, Testing sample %d...' % (len(train), len(test)))

    json.dump(train, open('data/vqa_raw_train.json', 'w'))
    json.dump(test, open('data/vqa_raw_test.json', 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', default=1, type=int,
                        help='Train on Train and test on Val, 2: train on Train+Val and test on Test')
    parser.add_argument('--dir', default=".", type=str,
                        help='The parent directory storing all VQAv2 related files and directories')

    args = parser.parse_args()
    params = vars(args)
    main(params)