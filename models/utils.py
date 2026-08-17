import torch, pickle
from torch.utils.data import Dataset
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def normalize_dataset(dataset, m_lat, m_lon):
    normalized_list = []
    for data, label in dataset:

        np_data = np.array(data[:600], dtype=np.float32)

        np_data[:, 5] = np_data[:, 5] / m_lat
        np_data[:, 6] = np_data[:, 6] / m_lon

        normalized_list.append([np_data.tolist(), label])
    return normalized_list


class Dataset_Seq_Img_with_PixelIndex(Dataset):
    def __init__(self, traj_init, map_13channel, map_6channel, pixel_index, class_list):
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
            new_index.append(seq_index[:600])
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
        
        # map image dataset: 13 channel
        map_13channel = np.array(map_13channel)
        self.map_13channel = torch.Tensor(map_13channel)

        # map extra img: 6 channel
        map_extra_6channel = np.array(map_6channel)
        self.map_extra_6channel = torch.Tensor(map_extra_6channel)

        # sample length
        self.n_samples = self.map_13channel.shape[0]

        # label
        self.label = torch.tensor([seq[1] for seq in traj_init])

        # class
        self.class_list = torch.tensor(class_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        map_img_sample = self.map_13channel[index]
        map_extra_sample = self.map_extra_6channel[index]
        label = self.label[index]
        init_traj_sample = self.clean_traj[index]
        pixel_index = self.pixel_index[index]
        class_label = self.class_list[index]

        return init_traj_sample, map_img_sample, map_extra_sample, label, class_label, pixel_index


def load_data_Seq_Img(args):

    if args.dataset == 'geo':
        traj_init_filename = './xx.pickle'
        map_filename = './xx.pickle'
        map_channel6_filename = './xx.pickle'
        # map image dataset: 13 channel
        with open(map_filename, "rb") as f:
            map_dataset = pickle.load(f)
        train_map_13channel, test_map_13channel = map_dataset
        # Index data
        traj_init_filename_geo = "./xx.pickle"
        with open(traj_init_filename_geo, "rb") as f:
            _, _, train_index, _, _, test_index = pickle.load(f) 
        filename_class_dist = './xx.pickle'

    # traj init dataset
    with open(traj_init_filename, "rb") as f:
        traj_dataset = pickle.load(f)
    train_init_traj, test_init_traj = traj_dataset

    # map extra img: 6 channel
    with open(map_channel6_filename, "rb") as f:
        map_extra_dataset = pickle.load(f)
    train_map_extra_6channel, test_map_extra_6channel = map_extra_dataset

    # class dist
    with open(filename_class_dist, "rb") as f:
        train_class, test_class = pickle.load(f)

    train_final = train_init_traj
    test_final = test_init_traj

    if args.Normalize_latlon:

        all_lats = []
        all_lons = []

        for sample in train_init_traj:

            if args.dataset == 'mtl':
                traj_data = sample[0]
                if len(sample[0]) > 600:

                    processed_traj = traj_data[:600]
                else:
                    processed_traj = traj_data
                sample[0] = processed_traj

            data_np = np.array(sample[0])
            all_lats.append(np.abs(data_np[:, 5]).max())
            all_lons.append(np.abs(data_np[:, 6]).max())

        max_lat = max(all_lats)
        max_lon = max(all_lons)

        train_norm = normalize_dataset(train_init_traj, max_lat, max_lon)
        test_norm = normalize_dataset(test_init_traj, max_lat, max_lon)

        train_final = train_norm
        test_final = test_norm

    train_merge_dataset = Dataset_Seq_Img_with_PixelIndex(train_final, train_map_13channel, train_map_extra_6channel, train_index, train_class)
    test_merge_dataset = Dataset_Seq_Img_with_PixelIndex(test_final, test_map_13channel, test_map_extra_6channel, test_index, test_class)

    train_loader = torch.utils.data.DataLoader(train_merge_dataset, shuffle=True, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_merge_dataset, shuffle=False, batch_size=args.batch_size)
    
    return train_loader, test_loader


class Dataset_init600_map19_index_class(Dataset):
    def __init__(self, traj_init, map13, map6, crop_map13, crop_map6, traj_index, class_list, traj_length): 
                #   crop_map13_5x5, crop_map6_5x5):
        # traj init dataset
        self.init_traj = traj_init        
        # traj img
        self.traj_map13 = map13
        self.traj_map6 = map6
        self.crop_map13 = crop_map13
        self.crop_map6 = crop_map6
        self.class_list = class_list
        # traj index
        self.traj_index = traj_index

        # sample length
        self.n_samples = len(traj_init)

        self.traj_len = traj_length

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        traj_map13 = self.traj_map13[index]
        traj_map6 = self.traj_map6[index]
        crop_map13 = self.crop_map13[index]
        crop_map6 = self.crop_map6[index]
        class_list = self.class_list[index]
        label = self.init_traj[index][1]

        new_traj_sample = []
        new_traj_index = []
        if len(self.init_traj[index][0]) < self.traj_len:
            # extra init traj
            extra = [[0 for j in range(len(self.init_traj[index][0][0]))] for i in range(self.traj_len - len(self.init_traj[index][0]))]
            new_traj_sample = self.init_traj[index][0] + extra
            # new_traj_sample = new_traj_sample[:600]
        else:
            new_traj_sample = self.init_traj[index][0]
            # new_traj_sample = new_traj_sample[:600]
        
        if len(self.traj_index[index][0]) < self.traj_len or len(self.traj_index[index][1]) < self.traj_len:
            # extra index
            extra_index = [[0 for i in range(self.traj_len - len(self.traj_index[index][0]))] for j in range(len(self.traj_index[index]))]
            extra_index_0 = self.traj_index[index][0] + extra_index[0]
            extra_index_1 = self.traj_index[index][1] + extra_index[1]
            new_traj_index.append(extra_index_0)
            new_traj_index.append(extra_index_1)
        else:
            new_traj_index = self.traj_index[index]

        init_traj_sample = torch.tensor(new_traj_sample)
        traj_map13 = np.array(traj_map13)
        traj_map13 = torch.Tensor(traj_map13)
        traj_map6 = np.array(traj_map6)
        traj_map6 = torch.Tensor(traj_map6)
        crop_map13 = np.array(crop_map13)
        crop_map13 = torch.Tensor(crop_map13)
        crop_map6 = np.array(crop_map6)
        crop_map6 = torch.Tensor(crop_map6)

        class_list = np.array(class_list)
        class_list = torch.Tensor(class_list)
        
        traj_index_lists = torch.tensor(np.array(new_traj_index))

        return init_traj_sample, traj_map13, traj_map6, crop_map13, crop_map6, traj_index_lists, class_list, label, index