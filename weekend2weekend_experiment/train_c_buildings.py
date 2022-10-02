"""
实验 1：
target building是 LIH
context转换场景是 ashrae 1 to 4
a: OIE
b: 除开LIH外的所有building
c: 由evaluation metrics筛选出来的最好的几个building
"""

from config import *
from GAN_model import Generator, Discriminator, TrainSet
from torch.utils.data import DataLoader

import pickle
import torch
import sys
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


def check_training_result(gen_a, gen_b, epoch):
    """
    Check generated data and best-match MAE after several epochs
    :param epoch
    :param gen_a
    :param gen_b
    :return: best-match MAE
    """
    mae_list = []
    for index, building_name in enumerate(c_building_names):
        with open('../tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(building_name, context_a, context_b),
                  'rb') as r:
            save_dict = pickle.load(r)

        # get domain a and domain b data
        context_a_data = save_dict.get('context_a_data')
        context_b_data = save_dict.get('context_b_data')
        load_max = save_dict.get('load_max')
        load_min = save_dict.get('load_min')

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
        print('test on: {}'.format(building_name))
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
        mae_list.append(cal_mae_for_best_match(generated_data=denorm_fake_data,
                                               target_data=denorm_real_data,
                                               is_weekend_flag=context_a_is_weekend))

        # check data generated
        # note: 这里只看在一个train building上test的效果
        if index == 0:
            flattened_fake_data = []
            for i in range(len(denorm_fake_data)):
                for j in range(len(denorm_fake_data[i])):
                    flattened_fake_data.append(denorm_fake_data[i][j])

            fig = plt.figure(figsize=(10, 6))
            fig.add_subplot(111)
            plt.plot(range(len(flattened_fake_data)), flattened_fake_data, color='b')
            plt.title("generated data (Epoch: {})".format(epoch))
            plt.savefig("{}/train_set_c_epoch_{}.png".format(train_result_save_path, epoch))

    mae = sum(mae_list) / len(mae_list)

    return mae

def train_GAN_on_c_buildings():
    """
    GAN training
    :return: gen_a & gen_b
    """
    cl_a_list = []
    cl_b_list = []
    for building_name in c_building_names:
        with open('../tmp_pkl_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(building_name, context_a, context_b),
                  'rb') as r:
            save_dict = pickle.load(r)
        this_cl_a = save_dict.get('cl_a')
        this_cl_b = save_dict.get('cl_b')
        cl_a_list.append(this_cl_a)
        cl_b_list.append(this_cl_b)
    data_a = np.concatenate(cl_a_list)
    data_b = np.concatenate(cl_b_list)

    # initialize trained_models
    gen_a = Generator()
    gen_b = Generator()
    dis_a = Discriminator()
    dis_b = Discriminator()

    # set up data loader
    train_dataset = TrainSet(data_a, data_b)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # set up parameters
    gen_params = list(gen_a.parameters()) + list(gen_b.parameters())
    dis_params = list(dis_a.parameters()) + list(dis_b.parameters())
    gen_opt = torch.optim.Adam(gen_params, lr=gen_learning_rate)
    dis_opt = torch.optim.Adam(dis_params, lr=dis_learning_rate)

    # start training
    best_gen_a = None
    best_gen_b = None
    min_mae = sys.maxsize
    for epoch in range(num_epochs):
        gen_loss_sum = 0
        dis_loss_sum = 0
        for i, (x_a, x_b) in enumerate(train_loader):
            x_a = x_a.to(torch.float32)
            x_b = x_b.to(torch.float32)

            # train discriminator
            dis_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_b, n_b = gen_b.encode(x_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ab = gen_b.decode(h_a + n_a)
            # discriminator loss
            loss_dis_a = dis_a.cal_dis_loss(x_ba, x_a)
            loss_dis_b = dis_b.cal_dis_loss(x_ab, x_b)
            loss_dis_all = loss_dis_a + loss_dis_b
            dis_loss_sum += loss_dis_all
            loss_dis_all.backward()
            dis_opt.step()

            # train generator
            gen_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_b, n_b = gen_b.encode(x_b)
            # decode (within domain)
            x_a_recon = gen_a.decode(h_a + n_a)
            x_b_recon = gen_b.decode(h_b + n_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ab = gen_b.decode(h_a + n_a)
            # encode again
            h_b_recon, n_b_recon = gen_a.encode(x_ba)
            h_a_recon, n_a_recon = gen_b.encode(x_ab)
            # decode again
            x_bab = gen_b.decode(h_b_recon + n_b_recon)
            x_aba = gen_a.decode(h_a_recon + n_a_recon)
            # generator loss
            # reconstruction loss
            loss_gen_recon_x_a = torch.mean(torch.abs(x_a_recon - x_a))
            loss_gen_recon_x_b = torch.mean(torch.abs(x_b_recon - x_b))
            # GAN loss
            loss_gen_adv_a = dis_a.cal_gen_loss(x_ba)
            loss_gen_adv_b = dis_b.cal_gen_loss(x_ab)
            # cycle-consistency loss
            loss_gen_cycle_cons_a = torch.mean(torch.abs(x_aba - x_a))
            loss_gen_cycle_cons_b = torch.mean(torch.abs(x_bab - x_b))
            # total loss
            loss_gen_all = loss_gen_recon_x_a + loss_gen_recon_x_b + \
                           loss_gen_adv_a + loss_gen_adv_b + \
                           loss_gen_cycle_cons_a + loss_gen_cycle_cons_b
            gen_loss_sum += loss_gen_all
            loss_gen_all.backward()
            gen_opt.step()

        if (epoch + 1) % check_every_n_epochs == 0:
            this_mae = check_training_result(gen_a, gen_b, epoch + 1)
            if this_mae < min_mae:
                best_gen_a = gen_a
                best_gen_b = gen_b
                min_mae = this_mae

        print('Epoch {}: Generator loss: {} \t Discriminator loss: {}'.format(epoch + 1, gen_loss_sum, dis_loss_sum))

    torch.save(best_gen_a.state_dict(), c_gen_a_save_path)
    torch.save(best_gen_b.state_dict(), c_gen_b_save_path)


if __name__ == '__main__':
    train_GAN_on_c_buildings()
