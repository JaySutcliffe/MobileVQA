import sys
import h5py
import numpy as np

from cnn import Feature_extracted_mobilenet_1by1
from data_generator import VQA_data_generator

# Set to false, problem with hdf files is lack of
# Java library support
hdf = False

# A file I used to generate mini packets of data to test on the Android application
# I tested using both h5 and binary files.
if __name__ == '__main__':
    vqa_gen = VQA_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=False,
                                 feature_object=
                                 Feature_extracted_mobilenet_1by1('D:/Part2Project/val30002.npy'))
    if hdf:
        for i in range(0, 20):
            [image_features, questions], answers = vqa_gen.__getitem__(i)
            f = h5py.File(sys.argv[1] + "/test_packet" + str(i) + ".h5", "w")
            f.create_dataset("ques", dtype='float32', data=questions)
            f.create_dataset("img_feats", dtype='float32', data=image_features)
            f.create_dataset("answers", dtype='int32', data=answers)
            f.close()
    else:
        for i in range(0, 20):
            [image_features, questions], answers = vqa_gen.__getitem__(i)
            floats = np.concatenate([questions.flatten(), image_features.flatten(),
                                     answers.flatten()]).astype(np.float32)
            floats.byteswap().tofile(sys.argv[1] + "/test_packet" + str(i) + ".bin")