import json
import re
import random
from shutil import copyfile

if __name__ == '__main__':
    location = "C:/Users/jaysu/AndroidStudioProjects/MVQA/app/src/main/assets/"
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