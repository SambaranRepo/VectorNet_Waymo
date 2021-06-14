import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from utils.waymo_motion_dataset2 import waymo_motion_dataset, collate

from utils.VectorNet import SubGraph, GNN
from utils.MTP_loss import multi_mode_loss_L2
from utils.Joint_metric import min_K_joint_ADE_metric
from utils.Decoder3 import Decoder

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
                        default = 100,
                        help = 'The number of epochs to train')
    parser.add_argument('--output_dir',
                        type = str,
                        default = './working_models_12_06_20',
                        help = 'The directory to save trained models')
    parser.add_argument('--lr',
                        type = float,
                        default = 1e-3,
                        help = 'Learning rate')
    parser.add_argument('--num_scenes',
                        type=int,
                        default=200,
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
                              num_workers = 4,
                              pin_memory = extras['pin_memory'])
    val_dataset = waymo_motion_dataset(dataroot = val_path_,
                                         scene_list = range(30))
    val_loader = DataLoader(dataset = val_dataset,
                              batch_size = parameters['batch_size'],
                              shuffle = True,
                              collate_fn = collate,
                              num_workers = 4,
                              pin_memory = extras['pin_memory'])


    # Define subgraph propagation networks
    N_hidden = 64
    # Agent vector is 9
    # [x, y, z, one-hot-type (5), id]
    agent_subgraph = SubGraph(8, N_hidden)
    # Roadmap vector is 27
    # [x, y, z, dir_x, dir_y, dir_z, one-hot-type (20), id]
    roadmap_subgraph = SubGraph(26, N_hidden)
    # Define global interaction graph
    #global_graph = GNN(N_hidden)  
    
    # Define decoder
    N_modes = 3
    T = 80
    decoder = Decoder(N_hidden + 5, N_modes, T)

    agent_subgraph = agent_subgraph.to(device)
    roadmap_subgraph = roadmap_subgraph.to(device)
    decoder = decoder.to(device)
    
    trainable_params = list(agent_subgraph.parameters()) \
                        + list(decoder.parameters())

    # Define loss functions, metrics, and optimizer
    optimizer = torch.optim.Adam(trainable_params, lr = parameters['lr'])

    for epoch in range(parameters['num_epochs']):
        Loss = 0
        if epoch%5==0: 
            total_val_loss = 0
            for minibatch_count, data in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    agents_polylines, agents_polylines_mask, \
                    roadgraph_polylines, roadgraph_polylines_mask, \
                    targets, targets_mask,\
                    tracks_to_predict, objects_of_interest,\
                    agents_motion = data

                    batch_size, num_agents, num_steps, dim_agent_vec = agents_polylines.shape
                    _, num_max_roads, num_max_road_length, dim_road_vec = roadgraph_polylines.shape

                    agents_polylines, agents_polylines_mask = agents_polylines.to(device), agents_polylines_mask.to(device)
                    roadgraph_polylines, roadgraph_polylines_mask = roadgraph_polylines.to(device), roadgraph_polylines_mask.to(device)
                    targets, targets_mask = targets.to(device), targets_mask.to(device)
                    agents_motion = agents_motion.to(device)

                    agent_polyline_features = agent_subgraph(agents_polylines[:, :, :, :8],
                                                             agents_polylines_mask,
                                                             agents_polylines.shape[1], agents_polylines.shape[2])
                    
                    roadgraph_polyline_features = roadmap_subgraph(roadgraph_polylines[:, :, :, :26], 
                                                                   roadgraph_polylines_mask,
                                                                   roadgraph_polylines.shape[1], roadgraph_polylines.shape[2])
                    

                    agent_feature_mask = torch.any(agents_polylines_mask, dim = 2)
                    #roadgraph_feature_mask = torch.any(roadgraph_polylines_mask, dim = 2)

                    graph_output = torch.cat([agent_polyline_features,roadgraph_polyline_features],axis = 1)
                    loss = 0

                    # Decode and compute loss scene by scene
                    for s in range(batch_size):
                        agent_inds = torch.nonzero((objects_of_interest == 1)[s,:], as_tuple=True)[0]
                        num_tracks_to_predict = agent_inds.shape[0]

        #                 for ind in agent_inds:
        # #                     target_features = agent_polyline_features[s, [ind.item()], :]
        #                     target_features = torch.cat([agent_polyline_features[s, [ind.item()], :], 
        #                                                  agents_motion[s, [ind.item()], :]], dim = 1)
        #     #                 target_features = torch.cat([graph_output[s,agent_inds,:], 
        #     #                                              agent_polyline_features[s, agent_inds, :]], dim = 1)
        #     #                 target_features = graph_output[s, agent_inds, :]
        #     #                 target_features = torch.cat(trainable_params = list(agent_subgraph.parameters()) \
        #     #                                              agents_polylines[s, agent_inds, 0, : 2],
        #     #                                              agents_motion[s, agent_inds, :]], dim = 1)
        #                     decoded_traj, decoded_probs = decoder(target_features)
        #                     ground_truth = targets[s, [ind.item()], :, :2]

        #                     pos_loss, log_loss, pos_loss_10 = multi_mode_loss_L2(decoded_traj, decoded_probs, ground_truth)
        #                     loss = loss + pos_loss + log_loss
        #                     pos_loss_total += pos_loss
        #                     pos_loss_10_total += pos_loss_10

        #                 target_features = agent_polyline_features[s, agent_inds, :]
        #                 target_features = torch.cat([agent_polyline_features[s, agent_inds, :], 
        #                                              agents_motion[s, agent_inds, :]], dim = 1)
        #                 target_features = torch.cat([agent_polyline_features[s, agent_inds, :], 
        #                                              agents_polylines[s, agent_inds, 0, : 2],
        #                                              agents_motion[s, agent_inds, :]], dim = 1)
        #                target_features = torch.cat([graph_output[s,agent_inds,:], 
        #                                              agent_polyline_features[s, agent_inds, :]], dim = 1)
        #                target_features = graph_output[s, agent_inds, :]
                        target_features = torch.cat([graph_output[s, agent_inds, :], 
                                                      agents_polylines[s, agent_inds, 0, : 2],
                                                      agents_motion[s, agent_inds, :]], dim = 1)
                        decoded_traj, decoded_probs = decoder(target_features)
                        ground_truth = targets[s,agent_inds,:,0:2]

                        loss,entropy_loss = multi_mode_loss_L2(decoded_traj, decoded_probs, ground_truth)
                        loss = loss + entropy_loss



                    loss = loss / batch_size / num_tracks_to_predict


                    total_val_loss+=loss
            print("Validation loss", total_val_loss / (len(val_dataset) / parameters['batch_size']))

        if epoch % 10== 0 and epoch != 0:
            for g in optimizer.param_groups:
                print('reduce')
                g['lr'] *= 0.33
        
        for minibatch_count, data in enumerate(tqdm(train_loader)):
            
            agents_polylines, agents_polylines_mask, \
            roadgraph_polylines, roadgraph_polylines_mask, \
            targets, targets_mask,\
            tracks_to_predict, objects_of_interest,\
            agents_motion = data

            batch_size, num_agents, num_steps, dim_agent_vec = agents_polylines.shape
            _, num_max_roads, num_max_road_length, dim_road_vec = roadgraph_polylines.shape
            
            agents_polylines, agents_polylines_mask = agents_polylines.to(device), agents_polylines_mask.to(device)
            roadgraph_polylines, roadgraph_polylines_mask = roadgraph_polylines.to(device), roadgraph_polylines_mask.to(device)
            targets, targets_mask = targets.to(device), targets_mask.to(device)
            agents_motion = agents_motion.to(device)
            
            agent_polyline_features = agent_subgraph(agents_polylines[:, :, :, :8],
                                                     agents_polylines_mask,
                                                     agents_polylines.shape[1], agents_polylines.shape[2])
            
            roadgraph_polyline_features = roadmap_subgraph(roadgraph_polylines[:, :, :, :26], 
                                                           roadgraph_polylines_mask,
                                                           roadgraph_polylines.shape[1], roadgraph_polylines.shape[2])
            

            agent_feature_mask = torch.any(agents_polylines_mask, dim = 2)
            #roadgraph_feature_mask = torch.any(roadgraph_polylines_mask, dim = 2)
            
            graph_output = torch.cat([agent_polyline_features,roadgraph_polyline_features],axis = 1)
            loss = 0
            
            # Decode and compute loss scene by scene
            for s in range(batch_size):
                agent_inds = torch.nonzero((objects_of_interest == 1)[s,:], as_tuple=True)[0]
                num_tracks_to_predict = agent_inds.shape[0]
    
#                 for ind in agent_inds:
# #                     target_features = agent_polyline_features[s, [ind.item()], :]
#                     target_features = torch.cat([agent_polyline_features[s, [ind.item()], :], 
#                                                  agents_motion[s, [ind.item()], :]], dim = 1)
#     #                 target_features = torch.cat([graph_output[s,agent_inds,:], 
#     #                                              agent_polyline_features[s, agent_inds, :]], dim = 1)
#     #                 target_features = graph_output[s, agent_inds, :]
#     #                 target_features = torch.cat(trainable_params = list(agent_subgraph.parameters()) \
#     #                                              agents_polylines[s, agent_inds, 0, : 2],
#     #                                              agents_motion[s, agent_inds, :]], dim = 1)
#                     decoded_traj, decoded_probs = decoder(target_features)
#                     ground_truth = targets[s, [ind.item()], :, :2]

#                     pos_loss, log_loss, pos_loss_10 = multi_mode_loss_L2(decoded_traj, decoded_probs, ground_truth)
#                     loss = loss + pos_loss + log_loss
#                     pos_loss_total += pos_loss
#                     pos_loss_10_total += pos_loss_10
            
#                 target_features = agent_polyline_features[s, agent_inds, :]
#                 target_features = torch.cat([agent_polyline_features[s, agent_inds, :], 
#                                              agents_motion[s, agent_inds, :]], dim = 1)
#                 target_features = torch.cat([agent_polyline_features[s, agent_inds, :], 
#                                              agents_polylines[s, agent_inds, 0, : 2],
#                                              agents_motion[s, agent_inds, :]], dim = 1)
#                target_features = torch.cat([graph_output[s,agent_inds,:], 
#                                              agent_polyline_features[s, agent_inds, :]], dim = 1)
#                target_features = graph_output[s, agent_inds, :]
                target_features = torch.cat([graph_output[s, agent_inds, :], 
                                              agents_polylines[s, agent_inds, 0, : 2],
                                              agents_motion[s, agent_inds, :]], dim = 1)
                decoded_traj, decoded_probs = decoder(target_features)
                ground_truth = targets[s,agent_inds,:,0:2]

                loss,entropy_loss = multi_mode_loss_L2(decoded_traj, decoded_probs, ground_truth)
                loss = loss + entropy_loss
               
                
         
            loss = loss / batch_size / num_tracks_to_predict
         
            
            Loss+=loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
#             print("Current loss", loss.item())
#             print("Current position loss", pos_loss_total.item())
#             print("Current position loss (10)", pos_loss_10_total.item())
       
            torch.save({'agent_subgraph':agent_subgraph.state_dict(),'roadmap_subgraph':roadmap_subgraph.state_dict(),
                        'decoder':decoder.state_dict()}, 
                       os.path.join(output_dir_, 'model_agent_roadmap_subraph_only.pt'))
        
        print('Epoch: {} Epoch Loss: {}'.format(epoch,Loss / (len(train_dataset) / parameters['batch_size'])))
            
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
