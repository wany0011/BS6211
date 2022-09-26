#ip_dir = '/home/chentyt/Documents/4tb/Tiana/P100/RCA_Split1/'
#op_dir = '/home/chentyt/Documents/4tb/Tiana/P100/P100_RCA/'

# Dataset:
dataset = 'batch1_2Frame'
#dataset = 'batch1_10Frame'

ip_dir = './sample/'+dataset+'/'
op_dir = './result/data/'+dataset+'/'
op_npz_dir = './result/data/'+dataset+'/npz/'
op_png_dir = './result/data/'+dataset+'/png/'


log_file = './result/log.txt'

seed = 17322625
epoch_no = 5000
patience = 100
lr = 1e-4

model_id = 'model0'
n_channel_2d = 32
n_downsampling = 2
out_kernel = 5
in_kernel = 5

neuron_list_b4_t = []
neuron_list_after_t = [512, 256, 128, 64, 32, 32]

batch_size = 32
num_pts = 128
img_loss_w = 0.1

# 'Res' or anything else
encoder = 'ResBN'

arch = 'C{}D{}O{}I{}FL{}_{}'.format(
    n_channel_2d,
    n_downsampling,
    out_kernel,
    in_kernel,
    len(neuron_list_b4_t),
    len(neuron_list_after_t)
    )

load_weight = True
