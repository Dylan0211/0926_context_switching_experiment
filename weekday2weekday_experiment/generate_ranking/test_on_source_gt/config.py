temperature_max = 40
temperature_min = 0

seq_length = 24
input_dim = 33
sparse_output_dim = 64
hidden_dim = 128
output_dim = 1
num_layers = 1

enc_hid_dim = 128
dec_hid_dim = 128
dropout = 0.5

sparse_lstm_lambda = 1e-3
sparse_ed_lambda = 1e-5

train_ratio = 0.5
batch_size = 64
learning_rate = 0.001
num_epochs = 10

model_structure_names = ['lstm', 'sparse_lstm', 'sparse_ed', 'seq2seq_with_attention']
building_names = ['CP4', 'DOH', 'OIE']
season_names = ['spring', 'summer']
model_name_list = ['{}_{}_{}'.format(model_structure_name, season_name, building_name)
                   for model_structure_name in model_structure_names
                   for season_name in season_names
                   for building_name in building_names]

# note: change the following parameters
# note: run "generate_data_using_GAN.py" to generate target context data
# note: run "test_all_models_on_generated_data.py" to generate a ranking of 24 models tested on the data
target_building_name = 'CP1'
source_context = 1
target_context = 4

