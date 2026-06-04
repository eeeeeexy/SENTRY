import torch
import torch.nn as nn

class GPS_Pixel_Align(nn.Module):
    def __init__(self, SECA_model, Img_model, Align_model, num_class, *args) -> None:
        super().__init__()
        self.traj_model = SECA_model
        self.img_model = Img_model
        self.align_model = Align_model

        # self.fc_traj = nn.Linear(128, 64)
        self.fc_img_map = nn.Linear(1600, 16)
        self.fc_merge = nn.Linear(128 + 128, num_class)

        self.fc_merge_layer = nn.Linear(128+128, 128)

        self.clean_output = nn.Linear(128, 32)
        self.adv_output = nn.Linear(128, 32)

    def forward(self, clean_traj=None, adv_traj=None, clean_map19=None, adv_map19=None, pixel_index=None, compressed_size=None, args=None):

        # 场景 B: 测试模式 (只传其中一个)
        target_traj = clean_traj if clean_traj is not None else adv_traj
        target_img = clean_map19 if clean_map19 is not None else adv_map19

        x_traj = self.traj_model(target_traj)
        x_img = self.img_model(target_img)
        x_emb = torch.cat((x_traj, x_img), dim=1)
        x_final = self.fc_merge(x_emb)

        reconstruction_seq, reconstruction_img = self.align_model(target_traj, target_img, pixel_index, compressed_size)
        x_re_traj = self.traj_model(reconstruction_seq)
        x_re_img = self.img_model(reconstruction_img)
        x_re_emb = torch.cat((x_re_traj, x_re_img), dim=1)
        x_re_final = self.fc_merge(x_re_emb)

        return x_traj, x_img, x_emb, x_final, x_re_final


