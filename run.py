import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from utils.waymo_motion_dataset_3 import waymo_motion_dataset, collate

from utils.VectorNet import SubGraph, GNN
from utils.MTP_loss import multi_mode_loss_L2
from utils.Joint_metric import min_K_joint_ADE_metric
from utils.Decoder import Decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VectorNet on Waymo motion dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_path', 
                        type = str,
                        default = '/datasets-2/waymo/uncompressed/tf_example/training',
                        help = 'The path that stores the training set of tf.Example format data')
    parser.add_argument('--val_path', 
                        type = str,
                        default = '/datasets-2/waymo/uncompressed/tf_example/validation_interactive',
                        help = 'The path that stores the validation set of tf.Example format data')
    parser.add_argument('--batch_size',
                        type = int,
                        default = 2,
                        help = 'The batch size of dataloader')
    parser.add_argument('--num_epochs',
                        type = int,
                        default = 10,
                        help = 'The number of epochs to train')
    parser.add_argument('--output_dir',
                        type = str,
                        default = './models',
                        help = 'The directory to save trained models')

    parameters = {}
    args = parser.parse_args()
    train_path_ = os.path.expanduser(args.train_path)
    val_path_ = os.path.expanduser(args.val_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    parameters['batch_size'] = args.batch_size
    parameters['num_epochs'] = args.num_epochs

    if not os.path.exists(output_dir_):
        os.mkdir(output_dir_)
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        extras = {"num_workers": len(os.sched_getaffinity(0)), "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        device = torch.device("cpu")
        extras = {"num_workers": len(os.sched_getaffinity(0)), "pin_memory": False}
        print("CUDA NOT supported")

    train_dataset = waymo_motion_dataset(dataroot = train_path_, scene_list = range(20))
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = parameters['batch_size'],
                              shuffle = False,
                              collate_fn = collate,
                              num_workers = 8,
                              pin_memory = extras['pin_memory'])


    # Define subgraph propagation networks
    N_hidden = 64
    # Agent vector is 5
    agent_feat = 5
    agent_subgraph = SubGraph(agent_feat, N_hidden)
    # Roadmap vector is 8
    roadmap_feat = 8
    roadmap_subgraph = SubGraph(roadmap_feat, N_hidden)
    # Define global interaction graph
    global_graph = GNN(2*N_hidden, N_hidden)

    # Define decoder
    N_modes = 3
    T = 80
    decoder = Decoder(N_hidden, N_modes, T)

    trainable_params = list(agent_subgraph.parameters()) \
                        + list(roadmap_subgraph.parameters()) \
                        + list(global_graph.parameters()) \
                        + list(decoder.parameters())

    # trainable_params = decoder.parameters()

    # Define loss functions, metrics, and optimizer
    loss_fn = multi_mode_loss_L2

    optimizer = torch.optim.Adam(trainable_params,lr = 1e-2)
    
    for epoch in range(parameters['num_epochs']):
        Loss = 0
        for minibatch_count, data in enumerate(train_loader):

            traffic_lights_polylines, traffic_lights_polylines_mask, \
            agents_polylines, agents_polylines_mask, \
            roadgraph_polylines, roadgraph_polylines_mask, \
            targets, targets_mask,\
            tracks_to_predict, objects_of_interest = data

            agents_polylines_mask = agents_polylines_mask[:,:,:,None]
            roadgraph_polylines_mask = roadgraph_polylines_mask[:,:,:,None]

            agent_polyline_features = agent_subgraph(agents_polylines, agents_polylines_mask, 
                                                     agents_polylines.shape[1], agents_polylines.shape[2])

            roadgraph_polyline_features = roadmap_subgraph(roadgraph_polylines, roadgraph_polylines_mask, 
                                                           roadgraph_polylines.shape[1], roadgraph_polylines.shape[2])

            graph_input = torch.cat([agent_polyline_features,roadgraph_polyline_features],axis = 1)
            graph_output = global_graph(graph_input)

            loss = 0
            # Decode and compute loss scene by scene
            for s in range(parameters['batch_size']):
                agent_inds = torch.nonzero(objects_of_interest[s,:]==1, as_tuple=True)[0]
                target_features = graph_output[s,agent_inds,:]
                decoded_traj, decoded_probs = decoder(target_features)
                ground_truth = targets[s,agent_inds,0:T,0:2]
                ground_truth_mask = targets_mask[s,agent_inds,0:T]

                min_joint_ADE, crossEnt = multi_mode_loss_L2(decoded_traj, decoded_probs, ground_truth, ground_truth_mask)
#                 print("Min ADE loss:", min_joint_ADE.item())
#                 print("CrossEnt loss:", crossEnt.item())
                loss += min_joint_ADE + crossEnt
                Loss +=loss
            # Backpropagation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch:",epoch,"Batch:",minibatch_count,"Current loss", loss.item())
        print("Total epoch loss : {}".format(Loss))
        torch.save({'agent_subgraph':agent_subgraph.state_dict(),'roadmap_subgraph':roadmap_subgraph.state_dict(),
                    'global_graph':global_graph.state_dict(),'decoder':decoder.state_dict()}, 
                       os.path.join(output_dir_, 'model.pt'))

#             if minibatch_count == 0:
#                 # [batch_size, num_traffic_lights num_steps, dim_traffic_lights_vec]
#                 print('traffic_lights_polylines.shape', traffic_lights_polylines.shape)
#                 # [batch_size, num_traffic_lights, num_steps]
#                 print('traffic_lights_polylines_mask.shape', traffic_lights_polylines_mask.shape)
#                 # [batch_size, num_agents, num_steps, dim_agent_vec]
#                 print('agents_polylines.shape', agents_polylines.shape)
#                 # [batch_size, num_agents, num_steps]
#                 print('agents_polylines_mask.shape', agents_polylines_mask.shape)
#                 # [batch_size, max_num_roads (varies for different batches), max_num_roads (varies for different batches), dim_roadgraph_vecs]
#                 print('roadgraph_polylines.shape', roadgraph_polylines.shape)
#                 # [batch_size, max_num_roads (varies for different batches), max_num_roads (varies for different batches)]
#                 print('roadgraph_polylines_mask.shape', roadgraph_polylines_mask.shape)
#                 # [batch_size, num_agents, num_future_steps, dim_future_vec]
#                 print('targets.shape', targets.shape)
#                 # [batch_size, num_agents, num_future_steps]
#                 print('targets_mask.shape', targets_mask.shape)
