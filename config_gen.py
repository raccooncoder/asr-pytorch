import json

config = dict(
    num_epochs = 50,
    batch_size = 64,
    learning_rate = 0.002,
    n_cats = 29,
    img_pad_len = 870,
    txt_pad_len = 200,
    txt_pad_idx = 28,
    ctc_blank = 27,
    melspec_sample_rate = 22050,
    melspec_n_mels = 64,
    melspec_n_fft = 1024,
    melspec_hop_length = 256,
    melspec_f_max = 8000,
    dataloader_num_workers = 8,
    seed = 13,
    aug_freq_mask_param = 30,
    aug_time_mask_param = 100,
    log_freq = 30,
    eval_freq = 10,
    dataset_path = 'LJSpeech-1.1/'
)

with open('config.json', 'w') as f:
    json.dump(config, f, indent=4)