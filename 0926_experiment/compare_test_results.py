"""
比较在a，b，c上训练的GAN在target building上的效果
"""

from config import *
from torch.utils.data import DataLoader
from GAN_model import TrainSet, Generator

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_data(context_a_data, context_b_data, gen_a, gen_b, load_max, load_min):
    data_a = []
    data_b = []
    if not context_a_is_weekend and not context_b_is_weekend:  # weekday to weekday
        temp_a = min([len(context_a_data[i]) for i in range(5)])
        temp_b = min([len(context_b_data[i]) for i in range(5)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(5):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    elif not context_a_is_weekend and context_b_is_weekend:  # weekday to weekend
        temp_a = min([len(context_a_data[i]) for i in range(2)])  # 周一周二 -> 周六周日
        temp_b = min([len(context_b_data[i]) for i in range(2)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(2):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    elif context_a_is_weekend and not context_b_is_weekend:  # weekend to weekday
        temp_a = min([len(context_a_data[i]) for i in range(5)])
        temp_b = min([len(context_b_data[i]) for i in range(5)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(5):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    else:  # weekend to weekend
        temp_a = min([len(context_a_data[i]) for i in range(2)])
        temp_b = min([len(context_b_data[i]) for i in range(2)])
        num_days = min(temp_a, temp_b)
        for i in range(num_days):
            for j in range(2):
                data_a.append(context_a_data[j][i])
                data_b.append(context_b_data[j][i])
    data_a = np.array(data_a)
    data_b = np.array(data_b)

    test_dataset = TrainSet(data_a, data_b)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print('test on: {}_ashrae_{}_to_{}_cl.pkl'.format(target_building_name, context_a, context_b))
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # start testing
    fake = []
    real = []
    original = []
    for x_a, x_b in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = gen_a.encode(x_a)
        output = gen_b.decode(content)
        fake.append(output.detach().numpy())
        real.append(x_b.detach().numpy())
        original.append(x_a.detach().numpy())

    # data processing
    fake_data = []
    real_data = []
    original_data = []
    for i in range(len(fake)):
        for j in range(fake[i].shape[0]):
            fake_data.append(fake[i][j])
            real_data.append(real[i][j])
            original_data.append(original[i][j])

    final_fake_data = []
    final_real_data = []
    final_original_data = []
    for i in range(len(real_data)):
        for j in range(real_data[i].shape[0]):
            final_real_data.append(real_data[i][j])
            final_fake_data.append(fake_data[i][j])
            final_original_data.append(original_data[i][j])

    # denormalize
    denormalized_fake_data = [final_fake_data[i] * (load_max - load_min) + load_min for i in range(len(final_fake_data))]
    denormalized_real_data = [final_real_data[i] * (load_max - load_min) + load_min for i in range(len(final_real_data))]
    denormalized_original_data = [final_original_data[i] * (load_max - load_min) + load_min for i in range(len(final_original_data))]

    # error
    mae_list = [abs(denormalized_real_data[i] - denormalized_fake_data[i]) for i in range(len(denormalized_real_data))]
    mae = sum(mae_list) / len(mae_list)

    # draw graphs
    # fig = plt.figure(figsize=(10, 6))
    # fig.add_subplot(111)
    # plt.plot(range(len(final_real_data)), final_real_data, label='real_data', color='blue')
    # plt.plot(range(len(final_fake_data)), final_fake_data, label='generated_data', color='red')
    # plt.plot(range(len(final_original_data)), final_original_data, label='original_data', color='green')
    # plt.title('MAE = {}'.format(mae), loc='right')
    # plt.grid()
    # plt.legend(loc=1, fontsize=15)
    # plt.show()

    return mae


def test_a_b_c_on_target_building():
    # load test data
    with open('../tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(target_building_name, context_a, context_b),
              'rb') as r:
        save_dict = pickle.load(r)
    context_a_data = save_dict.get('context_a_data')
    context_b_data = save_dict.get('context_b_data')
    load_max = save_dict.get('load_max')
    load_min = save_dict.get('load_min')

    # load models
    gen_a_on_a_building = Generator()
    gen_a_on_a_building.load_state_dict(torch.load('models/a_gen_a_cl.pt'))
    gen_b_on_a_building = Generator()
    gen_b_on_a_building.load_state_dict(torch.load('models/a_gen_b_cl.pt'))
    gen_a_on_b_buildings = Generator()
    gen_a_on_b_buildings.load_state_dict(torch.load('models/b_gen_a_cl.pt'))
    gen_b_on_b_buildings = Generator()
    gen_b_on_b_buildings.load_state_dict(torch.load('models/b_gen_b_cl.pt'))
    gen_a_on_c_buildings = Generator()
    gen_a_on_c_buildings.load_state_dict(torch.load('models/c_gen_a_cl.pt'))
    gen_b_on_c_buildings = Generator()
    gen_b_on_c_buildings.load_state_dict(torch.load('models/c_gen_b_cl.pt'))

    # start testing and get results
    mae_on_a_building = generate_data(context_a_data=context_a_data, context_b_data=context_b_data,
                                      gen_a=gen_a_on_a_building, gen_b=gen_b_on_a_building,
                                      load_max=load_max, load_min=load_min)
    mae_on_b_buildings = generate_data(context_a_data=context_a_data, context_b_data=context_b_data,
                                       gen_a=gen_a_on_b_buildings, gen_b=gen_b_on_b_buildings,
                                       load_max=load_max, load_min=load_min)
    mae_on_c_buildings = generate_data(context_a_data=context_a_data, context_b_data=context_b_data,
                                       gen_a=gen_a_on_c_buildings, gen_b=gen_b_on_c_buildings,
                                       load_max=load_max, load_min=load_min)

    # display results
    print('MAE of GAN on {} tested on {}: \t {}'.format(a_building_name, target_building_name, mae_on_a_building))
    print('MAE of GAN on {} tested on {}: \t {}'.format(b_building_names, target_building_name, mae_on_b_buildings))
    print('MAE of GAN on {} tested on {}: \t {}'.format(c_building_names, target_building_name, mae_on_c_buildings))


if __name__ == '__main__':
    test_a_b_c_on_target_building()
