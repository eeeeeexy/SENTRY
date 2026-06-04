import torch, pickle
from torch.utils.data import Dataset
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Dataset_Seq_Img_with_PixelIndex(Dataset):
    def __init__(self, traj_init, map_13channel, map_6channel, pixel_index):
        # traj init dataset
        # self.init_traj = traj_init
        self.clean_traj = [seq[0] for seq in traj_init]
    
        new_index = []
        for seq_p in pixel_index:
            seq_index = []
            for p_index, _ in enumerate(seq_p[0]):
                seq_index.append((seq_p[0][p_index], seq_p[1][p_index]))
            if len(seq_index) < 600:
                extra_index = [seq_index[-1] for _ in range(600 - len(seq_index))]
                seq_index += extra_index
            new_index.append(seq_index)
        new_index = np.array(new_index)
        self.pixel_index = torch.tensor(new_index)

        new_traj_sample = []
        for index, seq in enumerate(self.clean_traj):
            if len(seq) < 600:
                extra = [seq[-1] for i in range(600 - len(seq))]
                new_traj_sample.append(seq + extra)
            else:
                new_traj_sample.append(seq)
        self.clean_traj = torch.tensor(new_traj_sample)

        map_13channel = np.array(map_13channel)
        self.map_13channel = torch.Tensor(map_13channel)

        map_extra_6channel = np.array(map_6channel)
        self.map_extra_6channel = torch.Tensor(map_extra_6channel)

        self.n_samples = self.map_13channel.shape[0]
        self.label = torch.tensor([seq[1] for seq in traj_init])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        map_img_sample = self.map_13channel[index]
        map_extra_sample = self.map_extra_6channel[index]
        label = self.label[index]
        init_traj_sample = self.clean_traj[index]
        pixel_index = self.pixel_index[index]

        return init_traj_sample, map_img_sample, map_extra_sample, label, pixel_index



def normalize_dataset(dataset, m_lat, m_lon):
    normalized_list = []
    for data, label in dataset:
        np_data = np.array(data, dtype=np.float32)
        np_data[:, 5] = np_data[:, 5] / m_lat
        np_data[:, 6] = np_data[:, 6] / m_lon
        normalized_list.append([np_data.tolist(), label])
    return normalized_list


def load_data_Seq_Img(args):

    traj_init_filename = 'xxx.pickle'

    map_filename = 'xxx.pickle'
    map_channel6_filename = 'xxx.pickle'

    # traj init dataset
    with open(traj_init_filename, "rb") as f:
        traj_dataset = pickle.load(f)
    train_init_traj, test_init_traj = traj_dataset
    with open(map_filename, "rb") as f:
        map_dataset = pickle.load(f)
    train_map_13channel, test_map_13channel = map_dataset

    with open(map_channel6_filename, "rb") as f:
        map_extra_dataset = pickle.load(f)
    train_map_extra_6channel, test_map_extra_6channel = map_extra_dataset

    train_final = train_init_traj
    test_final = test_init_traj

    if args.Normalize_latlon:

        all_lats = []
        all_lons = []

        for sample in train_init_traj+test_init_traj:
            # sample[0] 是形状为 (600, 7) 的轨迹数据
            data_np = np.array(sample[0]) 
            all_lats.append(np.abs(data_np[:, 5]).max())
            all_lons.append(np.abs(data_np[:, 6]).max())

        max_lat = max(all_lats)
        max_lon = max(all_lons)

        train_norm = normalize_dataset(train_init_traj, max_lat, max_lon)
        test_norm = normalize_dataset(test_init_traj, max_lat, max_lon)

        train_final = train_norm
        test_final = test_norm

    traj_init_filename_geo = "xxx.pickle"
    with open(traj_init_filename_geo, "rb") as f:
        train_selfimg5, train_label, train_index, test_selfimg5, test_label, test_index = pickle.load(f) 
    train_merge_dataset = Dataset_Seq_Img_with_PixelIndex(train_final, train_map_13channel, train_map_extra_6channel, train_index)
    test_merge_dataset = Dataset_Seq_Img_with_PixelIndex(test_final, test_map_13channel, test_map_extra_6channel, test_index)

    train_loader = torch.utils.data.DataLoader(train_merge_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_merge_dataset, shuffle=False, batch_size=args.batch_size)
    
    return train_loader, test_loader

