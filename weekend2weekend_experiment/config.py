dis_learning_rate = 0.0001
gen_learning_rate = 0.001
num_epochs = 100
batch_size = 64
train_size = 0.5

temperature_min = 0
temperature_max = 40

check_every_n_epochs = 10

"""
winter peak weekday,                0
winter average weekday,             1
winter average weekend day/holiday, 2

summer peak weekday,                3
summer average weekday,             4
summer average weekend day/holiday, 5

spring average weekday,             6 
fall average weekday                7
"""

# note: change the following parameters
# note: run "train_a_building.py" to use train set a to generate a GAN
# note: run "train_b_buildings.py" to use train set b to generate a GAN
# note: run "train_c_buildings.py" to use train set c to generate a GAN
# note: run "compare_test_results.py" to compare MAE of the above three GANs on target building
context_a = 2
context_b = 5
context_a_is_weekend = True if context_a in [2, 5] else False
context_b_is_weekend = True if context_b in [2, 5] else False

# note: a = a single building
# note: b = all other buildings except target building
# note: c = "best set of buildings" selected by us
a_building_name = 'CPN'
b_building_names = ['CP4', 'CPN', 'CPS', 'DEH', 'DOH', 'OIE', 'OXH']
c_building_names = ['OIE', 'CP4']
target_building_name = 'CP1'

# note: GAN models save path
a_gen_a_save_path = './models/weekend2weekend_train_set_a_gen_a.pt'
a_gen_b_save_path = './models/weekend2weekend_train_set_a_gen_b.pt'
b_gen_a_save_path = './models/weekend2weekend_train_set_b_gen_a.pt'
b_gen_b_save_path = './models/weekend2weekend_train_set_b_gen_b.pt'
c_gen_a_save_path = './models/weekend2weekend_train_set_c_gen_a.pt'
c_gen_b_save_path = './models/weekend2weekend_train_set_c_gen_b.pt'

# note: train result img save path
train_result_save_path = './train_results_along_epochs'
