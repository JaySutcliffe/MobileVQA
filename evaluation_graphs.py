import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os


def kb_to_mb(arr):
    """
    Converts an array in MB to KB

    Parameters:
        arr (int array): array entered
    """
    for i in range(0, len(arr)):
        arr[i] = arr[i] / 1024


def size_bar_chart():
    """
    Creates bar charts to represent the various model sizes
    """
    base_sizes = [65726, 32693, 14429]
    kb_to_mb(base_sizes)
    pruned_sizes = [41351, 21463, 9447]
    kb_to_mb(pruned_sizes)
    attention_sizes = [309906, 152429, 76982]
    kb_to_mb(attention_sizes)

    x_names = ["Base", "F16", "Dynamic"]

    idxs = np.arange(len(base_sizes))

    width = 0.4
    
    plt.xlabel("Model Types")
    plt.ylabel("Model Size(MB)")
    plt.bar(idxs, base_sizes, width, label='Base')
    plt.bar(idxs + width, pruned_sizes, width, label='Pruned')

    plt.ylabel("Model Size (MB)")
    plt.title("Comparison Of Various Compressed Model Sizes")
    plt.xticks(idxs + width / 2, x_names)
    plt.legend(loc='best')
    plt.show()

    plt.xlabel("Model Types")
    plt.ylabel("Model Size(MB)")
    plt.bar(idxs, base_sizes, width, label='Base')
    plt.bar(idxs + width, attention_sizes, width, label='Attention', color='r')
    plt.ylabel("Model Size (MB)")
    plt.title("Comparison Of Base And Attention Model Sizes")
    plt.xticks(idxs + width / 2, x_names)
    plt.legend(loc='best')
    plt.show()


def box_plots(directory):
    """
    Creates box plots of all the results taken from device in a given directory

    Parameters:
        directory (string): the directory of the box plots
    """
    data = {"cnn_inference_times": [],
            "nlp_inference_times": [],
            "cpu_usages": []}
    entries = []
    for i, entry in enumerate(os.scandir(directory)):
        with open(entry.path) as data_file:
            entries.append(os.path.basename(entry).split('.')[0])
            item = json.load(data_file)
            data["cnn_inference_times"].append(item["cnn_inference_time"][60:200])
            data["nlp_inference_times"].append(item["nlp_inference_time"][60:200])
            data["cpu_usages"].append(item["cpu_usage"][60:200])
            print(entries[i])
            print("cnn: mean = ", np.mean(data["cnn_inference_times"][i]),
                  "std = ", np.std(data["cnn_inference_times"][i]))
            print("nlp: mean = ", np.mean(data["nlp_inference_times"][i]),
                  "std = ", np.std(data["nlp_inference_times"][i]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("CNN Inference Times")
    ax1.set_ylabel("Time (ms)")
    ax1.set_xlabel("Data Points")
    for i in range(0, len(data["cpu_usages"])):
        ax1.plot(np.arange(len(data["cnn_inference_times"][i])), np.array(data["cnn_inference_times"][i]),
                 label=entries[i])

    signs = np.array(data["nlp_inference_times"][0]) < np.array(data["nlp_inference_times"][1])
    print(np.count_nonzero(signs))

    ax2.set_title("VQA Part Inference Times")
    ax2.set_ylabel("Time (ms)")
    ax2.set_xlabel("Data Points")
    for i in range(0, len(data["cpu_usages"])):
        ax2.plot(np.arange(len(data["nlp_inference_times"][i])), np.array(data["nlp_inference_times"][i]),
                 label=entries[i])
    plt.legend()

    fig1, ax1 = plt.subplots()
    ax1.set_title("Inference Times For VQA Part")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Models")
    box = ax1.boxplot(data["nlp_inference_times"], vert=False,
                      flierprops={"marker": 'o',
                                  "markersize": 5,
                                  "markerfacecolor": "cornsilk"},
                      meanprops={"marker": "o",
                                 "markersize": 5,
                                 "markeredgecolor": "red",
                                 "markerfacecolor": "firebrick"},
                      patch_artist=True,
                      showmeans=True)

    for b in box["boxes"]:
        b.set_facecolor("wheat")

    ax1.xaxis.grid(True)
    plt.yticks(np.arange(1, 1+len(data["nlp_inference_times"])), entries)
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.set_title("Inference Times For MobileNet")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Models")
    box = ax1.boxplot(data["cnn_inference_times"], vert=False,
                      flierprops={"marker": 'o',
                                  "markersize": 5,
                                  "markerfacecolor": "cornsilk"},
                      meanprops={"marker": "o",
                                 "markersize": 5,
                                 "markeredgecolor": "red",
                                 "markerfacecolor": "firebrick"},
                      patch_artist=True,
                      showmeans=True)

    for b in box["boxes"]:
        b.set_facecolor("wheat")

    ax1.xaxis.grid(True)
    plt.yticks(np.arange(1, 1+len(data["cnn_inference_times"])), entries)
    plt.show()


def main(input_json):
    """
    Displays the usage of a particular input file

    Parameters:
        input_json (str): json file
    """
    with open(input_json) as data_file:
        data = json.load(data_file)

    plt.plot(np.arange(len(data["cnn_inference_time"])), np.array(data["cnn_inference_time"]))
    plt.show()

    plt.plot(np.arange(len(data["nlp_inference_time"])), np.array(data["nlp_inference_time"]))
    plt.show()

    # Cutting off warmup inferences
    data["cpu_usage"] = data["cpu_usage"][100:]

    fig1, ax1 = plt.subplots()
    ax1.set_title('Inference times')
    ax1.boxplot([data['cnn_inference_time'], data['nlp_inference_time']], showfliers=False)
    plt.show()
    print(np.median(data['cnn_inference_time']))
    print(np.median(data['nlp_inference_time']))

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
    fig1, ax1 = plt.subplots()
    ax1.set_title("CPU Usage Across Subset")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Data Points")
    plt.imshow(arr, cmap='Reds', interpolation='nearest',
               extent=[0, max_length * 10, 0, len(data["cpu_usage"])],
               aspect="auto")
    plt.show()

    arr2 = np.zeros((10, max_length))
    for i in range(0, len(data["cpu_usage"])):
        for j in range(0, len(data["cpu_usage"][i])):
            index = 9 - int(arr[i][j] // 10)
            arr2[index][j] += 1

    fig1, ax1 = plt.subplots()
    ax1.set_title("Average CPU Usage Across Subset")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("CPU Usage (%)")
    plt.imshow(arr2, cmap='Purples', interpolation='nearest', extent=[1, max_length * 10, 0, 100], aspect="auto")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_directory', default='device_results',
                        help='directory of all files for box plots')
    parser.add_argument('--input_name', default='Base.json',
                        help='target file name')
    args = parser.parse_args()
    params = vars(args)
    size_bar_chart()
    box_plots(params['input_directory'])
    main(params['input_directory'] + "/" + params['input_name'])