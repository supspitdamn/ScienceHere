import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

df = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\УИРС\\SEM5\\filtered_robot_data.csv")

class MLP:

    def __init__(self, in1, out1, out2):

        self.l1 = nn.Linear(in1, out1)
        self.f1 = F.sigmoid()
        self.l2 = nn.Linear(out1, out2)
        self.f2 = F.sigmoid()
    
    def forward(self, vec):

        


        
