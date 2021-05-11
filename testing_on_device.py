import json
import re
import random
from shutil import copyfile

# File used to load subset of question image pairs to the Android app's asset folder specified
# in variable location. I did not make this a command line argument since it will never change when set-up.
if __name__ == '__main__':
    location = "C:/Users/.../AndroidStudioProjects/MVQA/app/src/main/assets/"
    with open("data/vqa_raw_test.json") as data_file:
        data = json.load(data_file)

    test_on_device = []
    random.shuffle(data)
    for i in range(0, 500):
        new_dict = {}
        new_dict["question"] = data[i]["question"]
        new_dict["img_name"] = re.split("/", data[i]["img_path"])[-1]
        copyfile(data[i]["img_path"], location + "images/" + new_dict["img_name"])
        new_dict["ans"] = data[i]["ans"]
        test_on_device.append(new_dict)

    json.dump(test_on_device, open(location + "vqa_device_test.json", 'w'))