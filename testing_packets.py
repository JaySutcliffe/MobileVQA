import sys
import h5py
import numpy as np

from data_generator import VQA_data_generator


hdf = False

if __name__ == '__main__':
    vqa_gen = VQA_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=False,
                                 train_cnn=False, feature_file='D:/Part2Project/val.npy',
                                 batch_size=20, shuffle=True)
    vqa_gen.on_epoch_end()
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