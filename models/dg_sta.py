from st_att_layer import *
import torch.nn as nn
import torch

class DG_STA(nn.Module):
    def __init__(self, num_classes=14, dp_rate=0.5):
        super(DG_STA, self).__init__()

        h_dim = 32
        h_num= 8

        self.input_map = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            LayerNorm(128),
            nn.Dropout(dp_rate),
        )
        #input_size, h_num, h_dim, dp_rate, time_len, domain
        self.s_att = ST_ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="spatial", time_len = 20)


        self.t_att = ST_ATT_Layer(input_size=128, output_size= 128,h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, domain="temporal", time_len = 20)

        self.cls = nn.Linear(128, num_classes)


    def forward(self, x):
        # input shape: [batch_size, time_len, joint_num, 3]

        time_len = x.shape[1]
        joint_num = x.shape[2]

        #reshape x
        x = x.reshape(-1, time_len * joint_num,3)

        #input map
        x = self.input_map(x)
        #spatal
        print(x.shape)
        x = self.s_att(x)
        #temporal
        print(x.shape)
        x = self.t_att(x)
        print(x.shape)

        x = x.sum(1) / x.shape[1]
        print(x.shape)
        pred = self.cls(x)
        return pred

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DG_STA()
    model = model.to(device)
    dummy_input = torch.randn(2, 20, 22, 3).to(device)
    output = model(dummy_input)