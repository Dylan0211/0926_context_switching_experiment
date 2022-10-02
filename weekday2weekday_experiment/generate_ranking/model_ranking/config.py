# note: change the following parameters

# note: before run "compare_rankings.py" to calculate correlation coefficient between three rankings make sure you have
# note: run "generate_data_using_GAN.py" and "test_all_models_on_generated_data.py" under "test_on_generated_data",
# note: "test_on_source_gt" and "test_on_target_gt"
target_building = 'CP1'
GAN_building = 'c'
source_context = 1
target_context = 4
k = 10  # top-k

target_gt_rank_name = '../test_on_target_gt/test_result/test_on_{}_ashrae_{}_gt_result.csv'.format(
    target_building, target_context
)
source_gt_rank_name = '../test_on_source_gt/test_result/test_on_{}_ashrae_{}_gt_result.csv'.format(
    target_building, source_context
)
generated_data_rank_name = '../test_on_generated_data/test_result/test_on_{}_with_{}_GAN_ashrae_{}_to_{}_result.csv'.format(
    target_building, GAN_building, source_context, target_context
)

