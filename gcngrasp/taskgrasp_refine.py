import argparse
import os
import tqdm
import time
import copy
import sys
import pickle
import re

import pytorch_lightning as pl
import torch
import numpy as np
#from visualize import *

np.random.seed(10)
torch.manual_seed(10)

from networkx import convert_node_labels_to_integers
import torch.nn.functional as F
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

CODE_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(CODE_DIR)

DEVICE = "cuda"

import numpy as np
import torch
from torch_geometric.nn import SAGEConv, BatchNorm, GCNConv
import torch.nn.functional as F
import torch.nn as nn
import pickle
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from visualize import draw_scene, get_gripper_control_points
from data.GCNLoader import extract_subgraph
from networks.utils import transform_gripper_pc_custom

class GraphNet(torch.nn.Module):
    """Class for Graph Convolutional Network"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv_type):
        super(GraphNet, self).__init__()

        if conv_type == 'GCNConv':
            ConvLayer = GCNConv
        elif conv_type == 'SAGEConv':
            ConvLayer = SAGEConv
        else:
            raise NotImplementedError('Undefine graph conv type {}'.format(conv_type))

        # [B, 156, 301]
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(ConvLayer(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(ConvLayer(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        '''
        Inputs:
            x: [B, 156, 301]
            edge_index: [2, 1564]
        Outputs:
            x: [B, 156, 128]
        '''
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index) # [B, 156, 128]

            # make batch work! Only works when model.eval(True)
            x_size = x.size()
            x = x.reshape([x_size[0]*x_size[1], x_size[2]])
            x = batch_norm(x)
            x = x.reshape([x_size[0],x_size[1], x_size[2]])

            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

class GCNGrasp(torch.nn.Module):
    def __init__(self, embedding_size=300, gcn_num_layers=6, gcn_conv_type='GCNConv'):
        super().__init__()

        self.embedding_size = embedding_size
        self.gcn_num_layers = gcn_num_layers
        self.gcn_conv_type = gcn_conv_type

        self._build_model()

        # Load graph here
        graph_data_path = '/home/gpupc2/graspflow_models/graspflow_taskgrasp/data/knowledge_graph/kb2_task_wn_noi/graph_data.pkl'
    
        with open(graph_data_path, "rb") as fh:
            graph, seeds = pickle.load(fh)

        self.build_graph_embedding(graph)

    def _build_model(self):

        pc_dim = 1

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[pc_dim, 32, 32, 64], [pc_dim, 64, 64, 128], [pc_dim, 64, 96, 128]],
                use_xyz=True,
            )
        )

        input_channels = 64 + 128 + 128

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=True,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=True,
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.embedding_size)
        )

        self.fc_layer3 = nn.Sequential(
            # [1, 128]
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.gcn = GraphNet(
            in_channels=self.embedding_size+1,
            hidden_channels=128,
            out_channels=128,
            num_layers=self.gcn_num_layers,
            conv_type=self.gcn_conv_type)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, node_x_idx, latent, edge_index):
        """ Forward pass of GCNGrasp

        Args:
            pointcloud: Variable(torch.cuda.FloatTensor) [B, N, 4] tensor, 
                B is batch size, N is the number of points. The last channel is (x,y,z,feature)
            
            node_x_idx: [V*B] graph index used to lookup embedding dictionary, must be type long

            latent: tensor of size [V*B + B, 1] where V is size of the graph, used to indicate goal task and classes
            
            edge_index: graph adjaceny matrix of size [2, E*B], where E is the number of edges in the graph

        returns:
            logits: binary classification logits
        """

        xyz, features = self._break_up_pc(pointcloud)

        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
        shape_embedding = self.fc_layer(features.squeeze(-1)) # [1, 300]

        # node_x_idx = [155]
        node_embedding = self.graph_embedding(node_x_idx) # [155, 300]

        node_embedding = torch.cat([node_embedding, shape_embedding], dim=0) # [156, 300]
        # latent = [156, 1] [156, 300]
        # latent = [0...... 1, ...., 1,....., 1] <- 1 if task_gid, object_classs_gid or the last one

        node_embedding = torch.cat([node_embedding, latent], dim=1) # [156, 301]
        
        

        # Expected [B,156,128]
        node_embedding = node_embedding.unsqueeze(0)
        output = self.gcn(node_embedding, edge_index) # [156, 128]

        # Expected [B, 156,128]
        output = output.squeeze(0)

        batch_size = pointcloud.shape[0]
        output = output[-batch_size:, :] # [1, 128]
        logits = self.fc_layer3(output)

    
        return logits

    def build_graph_embedding(self, graph):
        """
        Creates and initializes embedding weights for tasks and class nodes in the graph.

        Args:
            graph: networkx DiGraph object
        """
        graph_size = len(list(graph.nodes))
        self.graph_embedding = nn.Embedding(graph_size, self.embedding_size)


def get_scores(translations, rotations, pc, intents, model):   
    # translations: [B, 3]
    # rotations: [B, 4]
    # pc: [B, 4096, 3]
    # intents: [B]
    
    pc = pc.cpu()
    B = translations.shape[0]

    # load graphs
    obj_class = 'spatula.n.01'
    with open('/home/gpupc2/graspflow_models/graspflow_taskgrasp/data/knowledge_graph/kb2_task_wn_noi/graph_data.pkl', "rb") as fh:
        graph, _ = pickle.load(fh)

    graph_idx = convert_node_labels_to_integers(graph)
    assert len(graph_idx.nodes) == len(graph.nodes)
    node_name2idx = {ent: idx for idx, ent in enumerate(list(graph.nodes))}

    torch.manual_seed(40)
    device = translations.device #torch.device('cuda:0')

    # differential robot gripper ---> taskgrasp gripper mapping
    # TODO YOU MUST
    
    rot_90_deg_mat = np.linalg.inv(R.from_euler('xz', [90, 90], degrees=True).as_matrix())
    rot_90_deg = R.from_matrix(rot_90_deg_mat).as_quat()
    mapping_rotations = torch.tensor(rot_90_deg, dtype=torch.float).to(device).repeat(B, 1)

    mapping_translations = torch.tensor([0, 0, 0.1-0.0032731392979622], dtype=torch.float).to(device).repeat(B, 1)

    # grasp_locations = transform_gripper_pc_custom(rotations, translations, torch.tensor(get_gripper_control_points(), dtype=torch.float).to(device), repeat=True)
    # grasp_locations = transform_gripper_pc_custom(mapping_rotations, mapping_translations, grasp_locations)
    
    grasp_locations = transform_gripper_pc_custom(mapping_rotations, mapping_translations, torch.tensor(get_gripper_control_points(), dtype=torch.float).to(device), repeat=True)
    grasp_locations = transform_gripper_pc_custom(rotations, translations, grasp_locations)
    

    #print(grasp_locations)
    x = torch.cat((pc.to(device), grasp_locations), dim = 1) # [B, 4096 + 7, 3]                                          
    
    # transformation_matrices = torch.eye(4).repeat(B, 1, 1)
    # rot_mat = R.from_quat(rotations.clone().detach().cpu().numpy()).as_matrix()

    # transformation_matrices[:, :3, :3] = torch.tensor(rot_mat)
    # transformation_matrices[:, :3, 3] = translations.clone().detach()

    # mapping_matrix = torch.eye(4).repeat(B, 1, 1)
    
    # mapping_matrix[:, :3, :3] = torch.tensor(rot_90_deg_mat).repeat(B, 1, 1)
    # mapping_matrix[:, :3, 3] = torch.tensor([0, 0, 0.1-0.0032731392979622], dtype=torch.float)
    
    # z = transformation_matrices @ mapping_matrix
    
    m = torch.max(torch.sqrt(torch.sum(x ** 2, dim=2)), dim=1)[0]
    m = torch.pow(m, -1)
    
    #print(x.shape)
    x = torch.cat([x, torch.ones(B, x.shape[1], 1).to(device)], dim=2)
    scales = []
    for i in m:
        scales.append(torch.diag(i.repeat(4)))
    scale_transform = torch.stack(scales)
    x = torch.matmul(scale_transform, torch.transpose(x, 2, 1))
    x = torch.transpose(x, 2, 1)
    x = x[:, :, :3]


    grasp_latent = torch.zeros((B, pc.shape[1] + grasp_locations.shape[1], 1))
    grasp_latent[:, -grasp_locations.shape[1]:, :] = 1

    x = torch.cat((x, grasp_latent.to(device)), dim = 2) # [B, 4096 + 7, 4]


    node_x_idx = torch.from_numpy(np.arange(155, dtype=int)).to(device)
    node_x_idx = node_x_idx.repeat(B)

    instance_gid = node_name2idx[obj_class]

    latents = []
    edges = []
    for i in range(len(intents)):
        task = intents[i]
        if task not in node_name2idx:
            task = "handover" # default to handover for unseen task
        task_gid = node_name2idx[task]

        # construct latent 
        latent = np.zeros([155]) # REPLACE
        latent[task_gid] = 1
        latent[instance_gid] = 1

        latents.append(torch.tensor(latent.reshape((latent.shape[0], 1))).to(device))

        G_s = extract_subgraph(graph_idx, instance_gid, task_gid)
        edge = np.array(list(G_s.edges)).T
        edge_src = np.expand_dims(edge[0, :], 0)
        edge_dest = np.expand_dims(edge[1, :], 0)
        edge_reverse = np.concatenate([edge_dest, edge_src], axis=0)
        edge = np.concatenate([edge, edge_reverse], axis=1)
        edges.append(edge)
    
    # pad all edges to the same length
    for i in range(len(edges)):
        edges[i] =  torch.tensor(edges[i])
        #padding = torch.tensor([edges[i][0, 0], edges[i][1, 0]]).view(2, 1).repeat(1, edge_max - edges[i].shape[1])
        padding = torch.tensor([])
        edges[i] = torch.cat([edges[i], padding], dim=1).to(device).long()


    latent = torch.cat(latents, dim=0).float()
    latent = torch.concat([latent, torch.ones([B,1]).to(device)], dim=0).float()

    edge_index = torch.cat(edges, dim=1).long()
    x = x.requires_grad_(True)
    out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent, edge_index=edge_index)
    logits = out.squeeze()
    probs = torch.sigmoid(logits)

    #visualize_taskgrasp(translations, rotations, pc, intents, probs)
    return logits, probs

def visualize_taskgrasp(translations, rotations, pc, intents, scores):
    # show grasps with pc
    B = translations.shape[0]
    rot_90_deg_mat = np.linalg.inv(R.from_euler('xz', [90, 90], degrees=True).as_matrix())
    mapping_translations = torch.tensor([0, 0.0, 0.1-0.0032731392979622], dtype=torch.float).numpy()

    rot_max = np.eye(4)
    rot_max[:3, :3] = rot_90_deg_mat
    rot_max[:3, 3] = mapping_translations
    grasps = []
    for i in range(len(translations)):
        g = np.eye(4)
        g[:3, :3] = R.from_quat(rotations[i].cpu()).as_matrix()
        g[:3, 3] = translations[i].cpu()
        g = g @ rot_max
        grasps.append(g)
        print("GRASPING")
        print(g @ get_gripper_control_points().T)


    
    grasps = np.array(grasps)


    draw_scene(pc.squeeze(0).cpu().numpy(), grasps=grasps)

    pass

def score(pc, node_x_idx, latent, edge_index):
    model = GCNGrasp()
    device = torch.device('cuda:0')
    torch.manual_seed(40)
    logits = model(
            pc.to(device),
            node_x_idx.to(device),
            latent.to(device),
            edge_index.to(device))
    
    logits = logits.squeeze()
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)

    return logits, probs

def evaluate(model, translations, rotations, pc, query):
    ## translations - [B, 3]
    ## rotations    - [B, 4]
    ## pc_mean      - [B, num_pc, 3]
    ## query        - [B]
    # print(translations.shape, rotations.shape, pc.shape, query.shape)
    logits, scores = get_scores(translations, rotations, pc, query, model)
    return logits, scores

def load_gcn_model(device=torch.device('cuda:0')):
    torch.manual_seed(40)
    # device = torch.device('cuda:0')

    # define model
    model = GCNGrasp()

    # load weights
    weights = torch.load('/home/gpupc2/graspflow_models/graspflow_taskgrasp/gcngrasp/gcn.pt')
    model.load_state_dict(weights)
    model = model.to(device)
    model = model.eval()  # make sure it's eval mode

    return model

if __name__ == "__main__":
    torch.manual_seed(40)

    #model = GraphNet(in_channels=301, hidden_channels=128, out_channels=128, num_layers=6, conv_type='GCNConv')

    device = torch.device('cuda:0')

    # define model
    model = GCNGrasp()

    # load weights
    weights = torch.load('/home/gpupc2/graspflow_models/graspflow_taskgrasp/gcngrasp/gcn.pt')
    model.load_state_dict(weights)
    model = model.to(device)
    model = model.eval()  # make sure it's eval mode


    #### TEST 1: NO BATCH ####
    print('Test 1')


    B = 1
    x = torch.rand([B, 4096+7,4]).to(device)
    node_x_idx = torch.from_numpy(np.arange(155, dtype=int)).to(device)
    node_x_idx = node_x_idx.repeat(B)
    latent = torch.rand([155, 1]).to(device)

    # we will use it for future
    latent_for_future = latent.repeat([4,1])
    latent_for_future = torch.concat([latent_for_future, torch.ones([4,1]).to(device)], dim=0)

    latent = latent.repeat([B,1])
    latent = torch.concat([latent, torch.ones([B,1]).to(device)], dim=0)

    edge_index = torch.rand([2, 1564]).to(device).long()

    edge_index_future = edge_index.repeat([1,4]) # optional

    print('IN')
    print(x.shape, node_x_idx.shape, latent.shape, edge_index.shape)
    out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent, edge_index=edge_index)

    print(out.sum(1))


    #### TEST 2: BATCH CASE ####
    # Output shall be 4 times replicated of the previous one
    print('Test 2')

    B = 4
    x = x.repeat([B,1,1])
    node_x_idx = node_x_idx.repeat(B)

    print('IN')
    print(x.shape, node_x_idx.shape, latent.shape, edge_index.shape)

    x = x.requires_grad_(True)
    
    out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent_for_future, edge_index=edge_index_future)

    loss = out.sum(1)

    print(loss)




    #### TEST 3: Backprop on input ####
    # You should see clear values in x.grad

    loss.backward(torch.ones_like(loss))

    print(x.grad)


    #### TEST 4: COMPARE with original one:
    # TODO
    print('Test 4')
    dddata = np.load("test_pc_base.npz")
    pc = torch.tensor(dddata["pc"])
    pc_mean = torch.tensor(dddata['pc_mean'])
    t = torch.tensor(dddata['t'])
    r = torch.tensor(dddata['r'])
    query = dddata['query']

    B = 4
    x = torch.rand([B, 4096+7,4]).to(device)
    x = x.repeat([B,1,1])
    latent = torch.rand([155, 1]).to(device)
    node_x_idx = torch.from_numpy(np.arange(155, dtype=int)).to(device)
    node_x_idx = node_x_idx.repeat(B)


    print('IN')
    print(x.shape, node_x_idx.shape, latent.shape, edge_index.shape)

    x = x.requires_grad_(True)
    
    out = model(pointcloud=x, node_x_idx=node_x_idx, latent=latent_for_future, edge_index=edge_index_future)

    loss = out.sum(1)

    exit()
