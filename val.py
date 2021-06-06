import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from utils.waymo_motion_dataset import waymo_motion_dataset, collate

from utils.VectorNet import SubGraph, GNN
from utils.MTP_loss import multi_mode_loss_L2
from utils.Joint_metric import min_K_joint_ADE_metric
from utils.Decoder import Decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VectorNet on Waymo motion dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
                        default = 1,
                        help = 'The batch size of dataloader')

    parameters = {}
    args = parser.parse_args()
    val_path_ = os.path.expanduser(args.val_path)
    parameters['batch_size'] = args.batch_size
    parameters['num_epochs'] = args.num_epochs

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        extras = {"num_workers": len(os.sched_getaffinity(0)), "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        device = torch.device("cpu")
        extras = {"num_workers": len(os.sched_getaffinity(0)), "pin_memory": False}
        print("CUDA NOT supported")

    val_dataset = waymo_motion_dataset(dataroot = val_path_, scene_list = range(150))
    val_loader = DataLoader(dataset = val_dataset,
                              batch_size = parameters['batch_size'],
                              shuffle = True,
                              collate_fn = collate,
                              num_workers = extras['num_workers'],
                              pin_memory = extras['pin_memory'])


    # Define subgraph propagation networks
    N_hidden = 64
    # Agent vector is 5
    agent_subgraph = SubGraph(5, N_hidden)
    # Roadmap vector is 8
    roadmap_subgraph = SubGraph(8, N_hidden)
    # Define global interaction graph
    global_graph = GNN(2*N_hidden,N_hidden)
    
    # Define decoder
    N_modes = 3
    T = 80
    decoder = Decoder(N_hidden, N_modes, T)
    model = torch.load('/home/sghosal/models/model.pt')
    agent_subgraph.load_state_dict(model['agent_subgraph'])
    roadmap_subgraph.load_state_dict(model['roadmap_subgraph'])
    global_graph.load_state_dict(model['global_graph'])
    decoder.load_state_dict(model['decoder'])
    MIN_ADE = 0
    for minibatch_count, data in enumerate(val_loader):

            traffic, traffic_masked, \
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
            min_ADE_metric = 0
            # Decode and compute loss scene by scene
            for s in range(parameters['batch_size']):
                agent_inds = torch.nonzero(objects_of_interest[s,:]==1, as_tuple=True)[0]
                target_features = graph_output[s,agent_inds,:]
                decoded_traj, decoded_probs = decoder(target_features)
                ground_truth = targets[s,agent_inds,0:T,0:2]
                ground_truth_mask = targets_mask[s,agent_inds,0:T]
                min_ADE_metric+=min_K_joint_ADE_metric(decoded_traj,ground_truth)
                min_ADE_metric/=parameters['batch_size']
            print("MIN_ADE for this batch was {}".format(min_ADE_metric))
            MIN_ADE+=min_ADE_metric
    MIN_ADE/75
    print("Total min_ade error is : {}".format(MIN_ADE))
