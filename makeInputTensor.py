from os import listdir
import tensorflow as tf
import numpy as np


def output_sequence(file_dir):
    with open(file_dir) as f:
        sequence = []
        lines = f.read()
        for line in lines.split('\n'):
            if not line == '':
                sequence.append(float(line))
    return sequence


def makeInputTensor(mode, dataset_path, data_type):
    data_type_dic = {
        "EDA": "EDA_microsiemens",
        "mmhg": "BP_mmHg",
        "mean": "LA Mean BP_mmHg",
        "sys": "LA Systolic BP_mmHg",
        "pulse": "Pulse Rate_BPM",
        "DIA": "BP Dia_mmHg",
        "volt": "Resp_Volts",
        "resp": "Respiration Rate_BPM"
    }
    if mode == "train":
        dataset_path += "/Training/"
    elif mode == "test":
        dataset_path += "/Testing/"
    elif mode == "valid":
        dataset_path += "/Validation/"
    sequence_list = []
    label_list = []
    for file_name in listdir(dataset_path):
        file_class = int(file_name.split('_')[1]) - 1
        one_hot = np.zeros(10, int)
        one_hot[file_class] = 1
        file_class_one_hot = one_hot
        file_data_type = file_name.split('_', 2)[2].split('.')[0]
        if data_type == "all":
            sequence_list.append(output_sequence(dataset_path + file_name))
            label_list.append(file_class_one_hot)
        elif file_data_type == data_type_dic[data_type]:
            sequence_list.append(output_sequence(dataset_path + file_name))
            label_list.append(file_class_one_hot)
    sequence_list = tf.keras.preprocessing.sequence.pad_sequences(sequence_list, padding="post", truncating="post",
                                                                  dtype=float, maxlen=992)
    label_list = np.array(label_list)
    return sequence_list, label_list
