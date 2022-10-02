dis_learning_rate = 0.0001
gen_learning_rate = 0.001
num_epochs = 20
batch_size = 64
train_size = 0.5

temperature_min = 0
temperature_max = 40

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

# data prepare and train
context_a = 2
context_b = 5
context_a_is_weekend = True if context_a in [2, 5] else False
context_b_is_weekend = True if context_b in [2, 5] else False

a_building_name = 'OIE'
b_building_names = ['CP4', 'CPN', 'CPS', 'DEH', 'DOH', 'OIE', 'OXH']
c_building_names = ['OIE', 'DEH']
target_building = 'CP1'

a_gen_a_save_path = './models/a_gen_a_cl.pt'
a_gen_b_save_path = './models/a_gen_b_cl.pt'
b_gen_a_save_path = './models/b_gen_a_cl.pt'
b_gen_b_save_path = './models/b_gen_b_cl.pt'
c_gen_a_save_path = './models/c_gen_a_cl.pt'
c_gen_b_save_path = './models/c_gen_b_cl.pt'
