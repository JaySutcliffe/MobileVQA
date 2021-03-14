import argparse
import matplotlib.pyplot as plt
import json

from evaluation.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
from evaluation.PythonHelperTools.vqaTools.vqa import VQA


version_type = 'v2_'
task_type = 'OpenEnded'
data_type = 'mscoco'
data_sub_type = 'val2014'
file_types = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']


def evaluate_results(annotation_json, question_json, results_json, result_type):
    [res_file, accuracy_file, eval_qa_file, eval_ques_type_file, eval_ans_type_file] = [
        'results/%s%s_%s_%s_%s_%s.json' % (version_type, task_type, data_type,
                                           data_sub_type, result_type, file_type)
        for file_type in file_types]

    # evaluate results
    vqa = VQA(annotation_json, question_json)
    vqa_res = vqa.loadRes(results_json, question_json)
    vqa_eval = VQAEval(vqa, vqa_res)
    vqa_eval.evaluate()

    # print accuracies
    print()
    print("Overall Accuracy is: %.02f\n" % (vqa_eval.accuracy['overall']))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqa_eval.accuracy['perQuestionType']:
        print("%s : %.02f" % (quesType, vqa_eval.accuracy['perQuestionType'][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqa_eval.accuracy['perAnswerType']:
        print("%s : %.02f" % (ansType, vqa_eval.accuracy['perAnswerType'][ansType]))
    print("\n")

    # plot accuracy for various question types
    plt.bar(range(len(vqa_eval.accuracy['perQuestionType'])), vqa_eval.accuracy['perQuestionType'].values(),
            align='center')
    plt.xticks(range(len(vqa_eval.accuracy['perQuestionType'])), vqa_eval.accuracy['perQuestionType'].keys(),
               rotation='0',
               fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.show()

    # save evaluation results to ./Results folder
    json.dump(vqa_eval.accuracy, open(accuracy_file, 'w'))
    json.dump(vqa_eval.evalQA, open(eval_qa_file, 'w'))
    json.dump(vqa_eval.evalQuesType, open(eval_ques_type_file, 'w'))
    json.dump(vqa_eval.evalAnsType, open(eval_ans_type_file, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotation_json', default='data/v2_mscoco_val2014_annotations.json',
                        help='the path to the json file')
    parser.add_argument('--question_json', default='data/v2_OpenEnded_mscoco_val2014_questions.json',
                        help='the path to the h5 file')
    parser.add_argument('--result_json', default='data/soft_test_results.json',
                        help='the path to the tne Tensorflow Lite mod')
    parser.add_argument('--result_type', default='soft_lstm_cnn',
                        help='the name of the model used')

    args = parser.parse_args()
    params = vars(args)

    evaluate_results(params['annotation_json'],
                     params['question_json'],
                     params['result_json'],
                     params['result_type'])