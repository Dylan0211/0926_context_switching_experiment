"""
目前的处理：
temp和coolingload是GAN生成的
其他时间特征是直接copy的用来生成的数据的时间特征
"""
from weekday2weekday_experiment.generate_ranking.dataloader import read_data
from config import *
from GAN_model import Generator, TrainSet
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt


def create_X_Y(df):
    X = []
    Y = []
    for i in range(df.shape[0] - seq_length):
        X.append(np.array(df.iloc[i: i + seq_length, 6:]))
        Y.append(df.loc[i + seq_length, 'nor_cl'])
    X = np.array(X)
    Y = np.array(Y).reshape(len(Y), 1)

    return X, Y


def generate_data(data_a, data_b):
    # set up trained_models
    gen_a = Generator()
    gen_b = Generator()
    gen_a.load_state_dict(torch.load(gen_a_save_path))
    gen_b.load_state_dict(torch.load(gen_b_save_path))

    test_dataset = TrainSet(data_a, data_b)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    print('data_a shape: {}'.format(data_a.shape))
    print('data_b shape: {}'.format(data_b.shape))

    # start testing
    fake = []
    for x_a, _ in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = gen_a.encode(x_a)
        output = gen_b.decode(content)
        fake.append(output.detach().numpy())

    # data processing
    fake_data = []
    for i in range(len(fake)):
        for j in range(fake[i].shape[0]):
            fake_data.append(fake[i][j])

    final_fake_data = []
    for i in range(len(fake_data)):
        for j in range(fake_data[i].shape[0]):
            final_fake_data.append(fake_data[i][j])

    return final_fake_data


def load_data():
    # load df
    df, load_max, load_min = read_data(building_name=target_building_name,
                                       model_name=target_building_name,
                                       seq_length=seq_length,
                                       temperature_min=temperature_min,
                                       temperature_max=temperature_max)

    # get source context data
    df = df[df['8_class'] == source_context]
    df.reset_index(drop=True, inplace=True)

    # generate ashrae 1 nor_cl
    data_a_cl = []
    for i in range(df.shape[0] // seq_length):
        data_a_cl.append(np.array(df.loc[seq_length * i: seq_length * (i + 1) - 1, 'nor_cl']))
    data_a_cl = np.array(data_a_cl)
    fake_nor_cl = generate_data(data_a_cl, data_a_cl)

    # create new df
    # note: 现在的温度是原始数据温度加10度（winter -> summer）
    df = df.loc[:len(fake_nor_cl) - 1, :]
    df['nor_cl'] = fake_nor_cl
    df['nor_temp'] = (df['temperature'] + temperature_diff - temperature_min) / (temperature_max - temperature_min)

    # create X and Y
    X, Y = create_X_Y(df=df)

    # save
    print('X shape: {}'.format(X.shape))
    print('Y shape: {}'.format(Y.shape))
    with open('./tmp_pkl_data/{}_ashrae_{}_to_{}_generated_data.pkl'.format(target_building_name, source_context, target_context), 'wb') as w:
        pickle.dump((X, Y, load_max, load_min), w)
    print('model saved: ./tmp_pkl_data/{}_ashrae_{}_to_{}_generated_data.pkl'.format(target_building_name, source_context, target_context))
    print()


if __name__ == '__main__':
    load_data()
