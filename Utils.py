import os
import random
import re

import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset

from draw_phase import draw_phase


def seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_file(file_name):
    with open(file_name, 'r') as file:
        return list((map(lambda l: list(map(float, l.split(','))), file.readlines())))


def parse_params(file_name):
    params = re.findall(r'([a-zA-Z]+)(\d+(\.\d+)?)', file_name)
    return dict(map(lambda p: (p[0], float(p[1])), params))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def partition(x_train, y_train, n):
    indexes = list(range(len(x_train)))
    random.shuffle(indexes)
    parts = [indexes[i::n] for i in range(n)]
    for p in parts:
        p.sort()
    print(parts)
    return [x_train[parts[i]] for i in range(n)], [y_train[parts[i]] for i in range(n)]


def split_x_train(x_train, y_train, test_size):
    # test_i = random.Random().sample(range(x_train.size()[0]), test_size)
    test_i = [random.randint(0, x_train.shape[0] - 1) for _ in range(test_size)]

    # test_i = [68, 66, 67, 33, 59, 65, 35, 12, 32, 46, 3, 42, 13, 73, 37, 5]

    # test_i = [0, 4, 7, 9, 12, 14, 17, 19, 23, 24, 50, 51, 58, 63, 68, 70]

    test_i.sort()
    print('test files', test_i)

    x_test = x_train[test_i]
    y_test = y_train[test_i]

    mask = np.ones(len(x_train), np.bool)
    mask[test_i] = 0
    new_x_train = x_train[mask]
    new_y_train = y_train[mask]

    return new_x_train, new_y_train, x_test, y_test


def convert_x_train(x_train, columns=None):
    x_train = np.array(x_train)

    t_count = len(np.unique(x_train[:, 0]))
    mu_count = len(np.unique(x_train[:, 1]))

    x_train = x_train[:t_count * mu_count]
    x_train = x_train.reshape((t_count, mu_count, -1))
    x_train = np.moveaxis(x_train, -1, 0)
    if not columns:
        columns = [2, 5, 9, 11, 12]
    x_train = x_train[columns]  # 3, 6, 10, 12, 13

    return x_train


def load_data(data_folder, keys, columns=None):
    x_train_list = []
    y_train_list = []
    for file_name in os.listdir(data_folder):
        params = parse_params(file_name)
        y_train_list.append(list(map(lambda k: params[k], keys)))
        x_train = convert_x_train(read_file(os.path.join(data_folder, file_name)), columns)
        x_train_list.append(x_train)
    return np.array(x_train_list), np.array(y_train_list)


def load_data_phase(data_folder, keys, size):
    x_train_list = []
    y_train_list = []

    params_strings = set()
    for file_name in os.listdir(data_folder):
        if 'result' in file_name:
            params_strings.add(file_name.split('_result')[0])

    for params_string in params_strings:
        params = parse_params(params_string)
        y_train_list.append(list(map(lambda k: params[k], keys)))
        x_max = 0.8
        y_max = 0.2
        phases = np.array([
            draw_phase(f'{data_folder}\\{params_string}_results_AFM_contour.dat', size, x_max, y_max),
            draw_phase(f'{data_folder}\\{params_string}_results_CO_contour.dat', size, x_max, y_max),
            draw_phase(f'{data_folder}\\{params_string}_results_FL_contour.dat', size, x_max, y_max),
            draw_phase(f'{data_folder}\\{params_string}_results_SC_contour.dat', size, x_max, y_max),
        ])

        # figure = plt.figure(figsize=(6, 6))
        # a = figure.add_subplot(1, 1, 1)
        palette = np.array([[255, 255, 255],
                            [255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0]], dtype=float)

        image = np.zeros((*size, 3), dtype=float)
        for i in range(4):
            image += palette[np.flip(phases[i].transpose(), 0) * (i + 1)] / 2

        # for i, phase in enumerate(phases):
        #     image += palette[np.flip(phase.transpose(), 0)]
        im = Image.fromarray(image.astype(np.uint8))
        im.save(f'{data_folder}\\{params_string}_plot.png')

        x_train_list.append(phases)
    x_train_list, y_train_list = np.array(x_train_list), np.array(y_train_list)
    print(len(x_train_list))
    # sort = y_train_list.argsort(axis=0).reshape((-1,))
    # return x_train_list[sort], y_train_list[sort]
    return x_train_list, y_train_list


def get_train_dataloader(x_train, y_train, batch_size):
    train_dataloader = torch.utils.data.DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                                                   num_workers=0)
    return train_dataloader


def get_test_dataloader(x_test, y_test, batch_size):
    test_dataloader = torch.utils.data.DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    return test_dataloader


def mean_absolute_percentage_error(preds, targets):
    return ((preds - targets) / targets).abs().mean()


def model_to_log(model):
    if type(model) is not nn.Sequential:
        return model
    return list(map(lambda s: model_to_log(s), model))
