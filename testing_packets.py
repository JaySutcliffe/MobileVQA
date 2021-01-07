import sys
import h5py

from data_generator import VQA_data_generator

if __name__ == '__main__':
    vqa_gen = VQA_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=False,
                                 train_cnn=False, feature_file='D:/Part2Project/val.npy',
                                 batch_size=1000, shuffle=True)
    vqa_gen.on_epoch_end()
    for i in range(0, 20):
        [image_features, questions], answers = vqa_gen.__getitem__(i)
        f = h5py.File(sys.argv[1]+"/test_packet"+str(i)+".h5", "w")
        f.create_dataset("ques", dtype='float32', data=questions)
        f.create_dataset("img_feats", dtype='float32', data=image_features)
        f.create_dataset("answers", dtype='int32', data=answers)
        f.close()