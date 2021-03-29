import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def main(input_json):
    with open(input_json) as data_file:
        data = json.load(data_file)

    plt.plot(np.arange(len(data['cnn_inference_time'])), np.array(data["cnn_inference_time"]))
    plt.show()

    plt.plot(np.arange(len(data['nlp_inference_time'])), np.array(data["nlp_inference_time"]))
    plt.show()

    # Cutting off warmup inferences
    data["cpu_usage"] = data["cpu_usage"][50:]

    # Finds maximum length for the plot
    max_length = 0
    for l in data["cpu_usage"]:
        if len(l) > max_length:
            max_length = len(l)

    arr = np.zeros((len(data["cpu_usage"]), max_length))
    for i in range(0, len(data["cpu_usage"])):
        for j in range(0, len(data["cpu_usage"][i])):
            arr[len(data["cpu_usage"]) - i - 1][j] = data["cpu_usage"][i][j]

    """
    Plots a heat-map. The colour demonstrates the CPU usage, the y-axis
    is each inference run and the x-axis is the inference times
    """
    plt.imshow(arr, cmap='Reds', interpolation='nearest',
               extent=[0, max_length * 10, 0, len(data["cpu_usage"])],
               aspect="auto")
    plt.show()

    arr2 = np.zeros((10, max_length))
    for i in range(0, len(data["cpu_usage"])):
        for j in range(0, len(data["cpu_usage"][i])):
            index = 9 - int(arr[i][j] // 10)
            arr2[index][j] += 1

    plt.imshow(arr2, cmap='Reds', interpolation='nearest', extent=[1, max_length * 10, 0, 100], aspect="auto")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='data/evaluation_output.json',
                        help='file containing index to word table')
    args = parser.parse_args()
    params = vars(args)
    main(params['input_json'])