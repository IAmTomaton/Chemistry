import argparse
import os

import numpy as np
import torch

from utils import read_phase_file
from model import load_model
from norm_denorm import load_mean_disp, denorm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean_disp_file = 'mean_disp.json'
mean, dispersion = load_mean_disp(mean_disp_file)

params = ['V', 'D', 'tp', 'tb']
models_files = [
    'models/size=64_params=V.pth',
    'models/size=64_params=D.pth',
    'models/size=64_params=tp.pth',
    'models/size=64_params=tb.pth',
]


def process_file(file_name):
    size = (4, 64, 64)
    models = [load_model(file, size, 1, device) for file in models_files]

    phases = read_phase_file(file_name)
    phases = torch.tensor(phases, dtype=torch.float, device=device)
    phases = phases.unsqueeze(0)

    results = []

    for model in models:
        result = model.forward(phases).detach().cpu().numpy()
        results.append(list(result[0])[0])

    results = denorm(np.array(results), mean, dispersion)
    results = list(results.astype(float))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--folder', type=str)
    parser.add_argument('--output_file', type=str)
    args_space = parser.parse_args()
    data_file = args_space.file
    data_folder = args_space.folder
    output_file = args_space.output_file

    if data_file is not None:
        results = process_file(data_file)
        if output_file is None:
            print('file', data_file)
            for param, result in zip(params, results):
                print(param, result)
        else:
            with open(output_file, 'w') as output:
                output.write(', '.join(params) + '\n')
                output.write(', '.join(map(str, results)) + '\n')

    elif data_folder is not None:
        full_results = []
        for data_file in os.listdir(data_folder):
            data_file = os.path.join(data_folder, data_file)
            results = process_file(data_file)
            full_results.append(results)
            if output_file is None:
                print('file', data_file)
                for param, result in zip(params, results):
                    print(param, result)
        if output_file is not None:
            with open(output_file, 'w') as output:
                output.write(', '.join(params) + '\n')
                for results in full_results:
                    output.write(', '.join(map(str, results)) + '\n')


if __name__ == '__main__':
    main()
