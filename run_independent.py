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
from utils.MTP_Loss2 import multi_mode_loss_L2
from utils.Joint_metric import min_K_joint_ADE_metric
from utils.Decoder2 import Decoder

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
                        default = 1,
                        help = 'The batch size of dataloader')
    parser.add_argument('--num_epochs',
                        type = int,
                        default = 150,
                        help = 'The number of epochs to train')
    parser.add_argument('--output_dir',
                        type = str,
                        default = './independent_prediction',
                        help = 'The directory to save trained models')
    parser.add_argument('--lr',
                        type = float,
                        default = 1e-4,
                        help = 'Learning rate')
    parser.add_argument('--num_scenes',
                        type=int,
                        default=100,
                        help='Number of scenes to be loaded')


    parameters = {}
    args = parser.parse_args()
    train_path_ = os.path.expanduser(args.train_path)
    val_path_ = os.path.expanduser(args.val_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    parameters['batch_size'] = args.batch_size
    parameters['num_epochs'] = args.num_epochs
    parameters['lr'] = args.lr
    parameters['num_scenes']=args.num_scenes

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

    train_dataset = waymo_motion_dataset(dataroot = train_path_,
                                         scene_list = range(parameters['num_scenes']))
    train_loader = DataLoader(dataset = train_dataset,
                              batch_size = parameters['batch_size'],
                              shuffle = True,
                              collate_fn = collate,
                              num_workers = 8,
                              pin_memory = extras['pin_memory'])


    # Define subgraph propagation networks
    N_hidden = 64
    # Agent vector is 3
    agent_subgraph = SubGraph(3, N_hidden).to(device)
    # Roadmap vector is 6
    roadmap_subgraph = SubGraph(6, N_hidden).to(device)
    # Define global interaction graph
    global_graph = GNN(N_hidden).to(device)
    
    # Define decoder
    N_modes = 1
    T = 80
    decoder = Decoder(N_hidden, N_modes, T).to(device)

    trainable_params = list(agent_subgraph.parameters()) \
                        + list(decoder.parameters())
    learn_rate = parameters['lr']
    
    # Define loss functions, metrics, and optimizer
    optimizer = torch.optim.Adam(trainable_params, lr = learn_rate)
    
    for epoch in range(parameters['num_epochs']):
        Loss = 0
        if epoch%5==0 and epoch!=0: 
            for g in optimizer.param_groups: 
                g['lr']*=0.33
            
        for minibatch_count, data in enumerate(tqdm(train_loader)):

            agents_polylines, agents_polylines_mask, \
            roadgraph_polylines, roadgraph_polylines_mask, \
            targets, targets_mask,\
            tracks_to_predict, objects_of_interest = data

            batch_size, num_agents, num_steps, dim_agent_vec = agents_polylines.shape
            _, num_max_roads, num_max_road_length, dim_road_vec = roadgraph_polylines.shape
            
            agent_polyline_features = agent_subgraph(agents_polylines.cuda(),
                                                     agents_polylines_mask.cuda(),
                                                     agents_polylines.shape[1], agents_polylines.shape[2])
            
            #roadgraph_polyline_features  =  roadmap_subgraph(roadgraph_polylines.cuda(),roadgraph_polylines_mask.cuda(),
                                                             #roadgraph_polylines.shape[1], roadgraph_polylines.shape[2])
            #graph_input = torch.cat([agent_polyline_features,roadgraph_polyline_features],axis = 1)
            graph_input = agent_polyline_features
            agent_feature_mask = torch.any(agents_polylines_mask, dim = 2)
            roadgraph_feature_mask = torch.any(roadgraph_polylines_mask, dim = 2)
            #graph_output = global_graph(input_var = graph_input.cuda(), 
                                        #input_mask = torch.cat([agent_features_mask.cuda(), roadgraph_features_mask.cuda()],axis = 1))
            graph_output = global_graph(input_var = graph_input.cuda(), 
                                        input_mask= agent_feature_mask.cuda())
             # Decode and compute loss scene by scene
            for s in range(parameters['batch_size']):
                # agent_inds = torch.nonzero(targets_mask[s,:,0], as_tuple=True)[0]
                agent_inds = torch.nonzero((objects_of_interest == 1)[s,:], as_tuple=True)[0]
                num_tracks_to_predict = agent_inds.shape[0]
                target_features = graph_output[s,agent_inds,:]
                decoded_traj = decoder(target_features)
                ground_truth = targets[s,agent_inds,:,0:2]
                
                loss1,loss2 = multi_mode_loss_L2(decoded_traj, ground_truth.cuda())
                
                
         
            loss1 = loss1 / batch_size 
            loss2 = loss2 / batch_size
            
            Loss+=loss1+loss2
            
            
            # Backpropagation
            optimizer.zero_grad()
            loss1.backward(retain_graph = True)
            
            loss2.backward()
            optimizer.step()
         
            print("Epoch: {} Batch:{} Current loss: {}".format(epoch,minibatch_count,loss1+loss2))
#             print("Current position loss", pos_loss_total.item())
#             print("Current position loss (10)", pos_loss_10_total.item())
       
            torch.save({'agent_subgraph':agent_subgraph.state_dict(), 'decoder':decoder.state_dict()},os.path.join(output_dir_, 'model.pt'))
        
        print('Epoch loss', Loss / (len(train_dataset) / parameters['batch_size']))
