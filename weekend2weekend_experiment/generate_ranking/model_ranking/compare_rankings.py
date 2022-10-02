from config import *
from model_rank_metrics_tool import jaccard_similarity_coefficient

import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv(target_gt_rank_name)
    target_gt_rank = df.loc[:, 'model_name'].tolist()

    df = pd.read_csv(source_gt_rank_name)
    source_gt_rank = df.loc[:, 'model_name'].tolist()

    df = pd.read_csv(generated_data_rank_name)
    generated_data_rank = df.loc[:, 'model_name'].tolist()

    # correlation coefficient
    corr_1 = jaccard_similarity_coefficient(target_gt_rank, source_gt_rank, k)
    corr_2 = jaccard_similarity_coefficient(target_gt_rank, generated_data_rank, k)
    print('train set: {}'.format(GAN_building))
    print('target building: {}'.format(target_building))
    print('Jaccard correlation coefficient between target context gt rank and source context gt rank is: {}'.format(corr_1))
    print('Jaccard correlation coefficient between target context gt rank and generated data rank is: {}'.format(corr_2))
