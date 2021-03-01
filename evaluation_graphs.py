import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    input_json = "data/evaluation.json"
    with open(input_json) as data_file:
        data = json.load(data_file)

    data["cpu_usage"] = data["cpu_usage"][1:]

    max_length = 0
    for l in data["cpu_usage"]:
        if len(l) > max_length:
            max_length = len(l)

    arr = np.zeros((len(data["cpu_usage"]), max_length))
    for i in range(0, len(data["cpu_usage"])):
        for j in range(0, len(data["cpu_usage"][i])):
            arr[len(data["cpu_usage"]) - i - 1][j] = data["cpu_usage"][i][j]

    cmap = plt.imshow(arr, cmap='Reds', interpolation='nearest', extent=[0, max_length * 10, 1, 100], aspect="auto")

    plt.show()

    arr2 = np.zeros((10, max_length))
    for i in range(0, len(data["cpu_usage"])):
        for j in range(0, len(data["cpu_usage"][i])):
            index = 9 - int(arr[i][j] // 10)
            arr2[index][j] += 1

    plt.imshow(arr2, cmap='Reds', interpolation='nearest', extent=[1, max_length * 10, 0, 100], aspect="auto")
    plt.show()
