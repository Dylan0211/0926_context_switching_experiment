"""
比较在a，b，c上训练的GAN在target building上的效果
"""

from config import *
from torch.utils.data import DataLoader
from GAN_model import TrainSet, Generator

import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt


def DTW_distance(arr_1, arr_2, max_warping_window=10000):
    """
    source: https://www.cnblogs.com/ningjing213/p/10502519.html
    :param arr_1
    :param arr_2
    :param max_warping_window
    :return: DTW distance: smaller => more similar
    """
    ts_a = np.array(arr_1)
    ts_b = np.array(arr_2)
    M = ts_a.shape[0]
    N = ts_b.shape[0]
    cost = np.ones((M, N))
    d = lambda x, y: ((x - y) ** 2)

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window), min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


def cal_mae_for_best_match(generated_data, target_data, is_weekend_flag):
    """
    for every 24h generated data, find best match (most similar) in ground truth of target context and calculate MAE
    :param generated_data
    :param target_data
    :param is_weekend_flag: whether source and target context is weekend
    :return: best-match MAE
    """
    num_days = 2 if is_weekend_flag else 5

    mae_list = []
    for i in range(len(generated_data)):
        this_data = generated_data[i]

        min_dtw_distance = sys.maxsize
        most_similar_data = None
        for j in range(len(target_data)):
            if j % num_days == i % num_days:
                possible_data = target_data[j]
                dtw_distance = DTW_distance(this_data, possible_data)
                if dtw_distance < min_dtw_distance:
                    min_dtw_distance = dtw_distance
                    most_similar_data = possible_data

        if most_similar_data is None:
            continue

        error_list = [abs(this_data[i] - most_similar_data[i]) for i in range(len(this_data))]
        mae_list.append(sum(error_list) / len(error_list))

    return sum(mae_list) / len(mae_list)


def test_data(context_a_data, context_b_data, gen_a, gen_b, load_max, load_min):
    """
    test data and calculate MAE
    :param context_a_data: data in context a
    :param context_b_data: data in context b
    :param gen_a: generator a
    :param gen_b: generator b
    :param load_max: max coolingLoad
    :param load_min: min coolingLoad
    :return: best-match MAE
    """
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
    for x_a, x_b in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = gen_a.encode(x_a)
        output = gen_b.decode(content)
        fake.append(output.detach().numpy())

    # data processing
    fake_data = []
    for i in range(len(fake)):
        for j in range(fake[i].shape[0]):
            fake_data.append(fake[i][j])

    # denormalize
    denorm_fake_data = []
    denorm_real_data = []
    for i in range(len(fake_data)):
        denorm_fake_data.append([fake_data[i][j] * (load_max - load_min) + load_min
                                 for j in range(fake_data[i].shape[0])])
        denorm_real_data.append([data_b[i][j] * (load_max - load_min) + load_min
                                 for j in range(data_b[i].shape[0])])

    # error
    mae = cal_mae_for_best_match(generated_data=denorm_fake_data,
                                 target_data=denorm_real_data,
                                 is_weekend_flag=context_a_is_weekend)

    return mae


def test_a_b_c_on_target_building():
    """
    Calculate best-match MAE for each a, b and c GAN model
    """
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
    gen_a_on_a_building.load_state_dict(torch.load('models/weekday2weekday_train_set_a_gen_a.pt'))
    gen_b_on_a_building = Generator()
    gen_b_on_a_building.load_state_dict(torch.load('models/weekday2weekday_train_set_a_gen_b.pt'))
    gen_a_on_b_buildings = Generator()
    gen_a_on_b_buildings.load_state_dict(torch.load('models/weekday2weekday_train_set_b_gen_a.pt'))
    gen_b_on_b_buildings = Generator()
    gen_b_on_b_buildings.load_state_dict(torch.load('models/weekday2weekday_train_set_b_gen_b.pt'))
    gen_a_on_c_buildings = Generator()
    gen_a_on_c_buildings.load_state_dict(torch.load('models/weekday2weekday_train_set_c_gen_a.pt'))
    gen_b_on_c_buildings = Generator()
    gen_b_on_c_buildings.load_state_dict(torch.load('models/weekday2weekday_train_set_c_gen_b.pt'))

    # start testing and get results
    mae_on_a_building = test_data(context_a_data=context_a_data, context_b_data=context_b_data,
                                  gen_a=gen_a_on_a_building, gen_b=gen_b_on_a_building,
                                  load_max=load_max, load_min=load_min)
    mae_on_b_buildings = test_data(context_a_data=context_a_data, context_b_data=context_b_data,
                                   gen_a=gen_a_on_b_buildings, gen_b=gen_b_on_b_buildings,
                                   load_max=load_max, load_min=load_min)
    mae_on_c_buildings = test_data(context_a_data=context_a_data, context_b_data=context_b_data,
                                   gen_a=gen_a_on_c_buildings, gen_b=gen_b_on_c_buildings,
                                   load_max=load_max, load_min=load_min)

    # display results
    print('MAE of GAN on {} tested on {}: \t {}'.format(a_building_name, target_building_name, mae_on_a_building))
    print('MAE of GAN on {} tested on {}: \t {}'.format(b_building_names, target_building_name, mae_on_b_buildings))
    print('MAE of GAN on {} tested on {}: \t {}'.format(c_building_names, target_building_name, mae_on_c_buildings))


if __name__ == '__main__':
    test_a_b_c_on_target_building()
