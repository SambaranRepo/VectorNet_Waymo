import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

def collate(samples):
    '''
    The collate function to support batch data loading
    '''
    agents_polylines, agents_polylines_mask, \
    roadgraph_polylines, roadgraph_polylines_mask, \
    targets, targets_mask, \
    tracks_to_predict, objects_of_interest, \
    agents_motion = map(list, zip(*samples))
    
    batch_size = len(roadgraph_polylines)
    max_num_roads = np.max([polyline.shape[0] for polyline in roadgraph_polylines])
    max_road_length = np.max([polyline.shape[1] for polyline in roadgraph_polylines])
    dim_roadgraph_vec = roadgraph_polylines[0].shape[2]

    roadgraph_polylines_padded = torch.zeros(batch_size, max_num_roads, max_road_length, dim_roadgraph_vec)
    roadgraph_polylines_mask_padded = torch.zeros(batch_size, max_num_roads, max_road_length, dtype=torch.bool)
    for batch_ind in range(batch_size):
        num_roads, road_length, _ = roadgraph_polylines[batch_ind].shape
        roadgraph_polylines_padded[batch_ind, : num_roads, : road_length, :] = roadgraph_polylines[batch_ind]
        roadgraph_polylines_mask_padded[batch_ind, : num_roads, : road_length] = roadgraph_polylines_mask[batch_ind]
    
    return torch.stack(agents_polylines), torch.stack(agents_polylines_mask), \
           roadgraph_polylines_padded, roadgraph_polylines_mask_padded, \
           torch.stack(targets), torch.stack(targets_mask), \
           torch.stack(tracks_to_predict), torch.stack(objects_of_interest), \
           torch.stack(agents_motion)

class waymo_motion_dataset(Dataset):
    
    def __init__(self,
                 dataroot,
                 scene_list = None):
        '''
        params:
        -- dataroot: The path that stores the tf.Example format data.
        -- scene_list: The indices of scene to load. If it is None, load all files under dataroot.
        '''
        
        self.dataroot = dataroot
        self.filenames = sorted(os.listdir(dataroot))
        
        if scene_list is None:
            self.num_scenes = len(self.filenames)
            self.scene_list = list(range(self.num_scenes))
        else:
            self.num_scenes = len(scene_list)
            self.scene_list = scene_list
        
        # Example field definition
        roadgraph_features = {
            'roadgraph_samples/dir':
                tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
            'roadgraph_samples/id':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/type':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/valid':
                tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
            'roadgraph_samples/xyz':
                tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
        }

        # Features of other agents.
        state_features = {
            'state/id':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/type':
                tf.io.FixedLenFeature([128], tf.float32, default_value=None),
            'state/is_sdc':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/tracks_to_predict':
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
            'state/current/bbox_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/height':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/length':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/timestamp_micros':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/valid':
                tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
            'state/current/vel_yaw':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/velocity_y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/width':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/x':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/y':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/current/z':
                tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
            'state/future/bbox_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/height':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/length':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/timestamp_micros':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/valid':
                tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
            'state/future/vel_yaw':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/velocity_y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/width':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/x':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/y':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/future/z':
                tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
            'state/past/bbox_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/height':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/length':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/timestamp_micros':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/valid':
                tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
            'state/past/vel_yaw':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/velocity_y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/width':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/x':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/y':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/past/z':
                tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
            'state/objects_of_interest': 
                tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        }

        traffic_light_features = {
            'traffic_light_state/current/state':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/valid':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/current/x':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/y':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/current/z':
                tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
            'traffic_light_state/past/state':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/valid':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/x':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/y':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/past/z':
                tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
            'traffic_light_state/current/id':
                tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
            'traffic_light_state/past/id':
                tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
            'traffic_light_state/past/state': 
                tf.io.FixedLenFeature([10,16],tf.int64,default_value=None),
            'traffic_light_state/current/state': 
                tf.io.FixedLenFeature([1,16],tf.int64,default_value=None),
            'traffic_light_state/past/timestamp_micros': 
                tf.io.FixedLenFeature([10],tf.int64,default_value=None),
            'traffic_light_state/current/timestamp_micros': 
                tf.io.FixedLenFeature([1],tf.int64,default_value=None),
        }

        self.features_description = {}
        self.features_description.update(roadgraph_features)
        self.features_description.update(state_features)
        self.features_description.update(traffic_light_features)
        
        self.data = [self.extract_data(ind) for ind in tqdm(self.scene_list)]
        
        self.filenames = [filename for (filename, x) in zip(self.filenames, self.data) if x is not None]
        self.data = [x for x in self.data if x is not None]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, ind):
        return self.data[ind]
    
    def extract_data(self, ind):
        filename = os.path.join(self.dataroot, self.filenames[ind])
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        data = next(dataset.as_numpy_iterator())
        parsed = tf.io.parse_single_example(data, self.features_description)
        
    
        # process past and current agent data        
        num_agents, num_past_steps = parsed['state/past/x'].shape
        _, num_current_steps = parsed['state/current/x'].shape
        
        agents_type_one_hot = np.eye(5)[parsed['state/type'].numpy().astype(np.int).flatten()].astype(np.float32)
#         print(agents_type_one_hot)
#         agents_type_one_hot = np.zeros((num_agents, 5))
#         agents_type_one_hot[np.arange(num_agents), parsed['state/type'].numpy().astype(np.int)] = 1
        
        past_agents = np.concatenate([parsed['state/past/x'].numpy()[:, :, None], 
                                 parsed['state/past/y'].numpy()[:, :, None], 
                                 parsed['state/past/z'].numpy()[:, :, None],
                                 np.tile(agents_type_one_hot[:, None, :], (1, num_past_steps, 1)),
                                 np.tile(parsed['state/id'].numpy().reshape(-1, 1, 1), (1, num_past_steps, 1))], axis = 2)
        current_agents = np.concatenate([parsed['state/current/x'].numpy()[:, :, None], 
                                 parsed['state/current/y'].numpy()[:, :, None], 
                                 parsed['state/current/z'].numpy()[:, :, None],
                                 np.tile(agents_type_one_hot[:, None, :], (1, num_current_steps, 1)),
                                 np.tile(parsed['state/id'].numpy().reshape(-1, 1, 1), (1, num_current_steps, 1))], axis = 2)
        
    
        
        agents_polylines = np.concatenate([current_agents, past_agents], axis = 1)
        agents_polylines_mask = np.concatenate([parsed['state/current/valid'],
                                                parsed['state/past/valid']], axis = 1).astype(np.bool)
        
        # process roadmap data
        num_roadgraph_samples = parsed['roadgraph_samples/type'].shape[0]
        roadgraph_type = parsed['roadgraph_samples/type'].numpy().astype(np.int).flatten()
        roadgraph_type_one_hot = np.eye(20)[roadgraph_type].astype(np.float32)
        
        roadgraph_id_set = set([ind for ind in parsed['roadgraph_samples/id'].numpy().reshape(-1).tolist() if ind > 0])
        roadgraph_vecs = np.concatenate([parsed['roadgraph_samples/xyz'],
                                         parsed['roadgraph_samples/dir'],
                                         roadgraph_type_one_hot,
                                         parsed['roadgraph_samples/id']], axis = 1)
        roadgraph_vecs_mask = parsed['roadgraph_samples/valid'].numpy().astype(np.bool).reshape(-1)
        
        # print(roadgraph_vecs.shape)
        
        num_points, dim_roadgraph_vecs = roadgraph_vecs.shape
        
        num_roads = len(roadgraph_id_set)
        max_road_length = np.max([np.sum(parsed['roadgraph_samples/id'].numpy().flatten() == ind) for ind in roadgraph_id_set])
        
        roadgraph_polylines = np.zeros(shape=(num_roads, max_road_length, dim_roadgraph_vecs))
        roadgraph_polylines_mask = np.zeros(shape=(num_roads, max_road_length), dtype=np.bool)
        
        for i, ind in enumerate(roadgraph_id_set):
            indices = parsed['roadgraph_samples/id'].numpy().flatten() == ind
            polyline_length = np.sum(indices)
            roadgraph_polylines[i, : polyline_length, :] = roadgraph_vecs[indices, :]
            roadgraph_polylines_mask[i, : polyline_length] = roadgraph_vecs_mask[indices]
        
        # process future (target) agent data
        _, num_future_steps = parsed['state/future/x'].shape
        targets = np.dstack([parsed['state/future/x'].numpy(), 
                             parsed['state/future/y'].numpy(), 
                             parsed['state/future/z'].numpy(),
                             np.tile(parsed['state/type'].numpy().reshape(-1, 1), (1, num_future_steps)),
                             np.tile(parsed['state/id'].numpy().reshape(-1, 1), (1, num_future_steps))])
        targets_mask = parsed['state/future/valid'].numpy().astype(np.bool)

        tracks_to_predict = parsed['state/tracks_to_predict'].numpy()
        objects_of_interest = parsed['state/objects_of_interest'].numpy()
        
        # process veocity, yaw
        agents_motion = np.concatenate([parsed['state/current/velocity_x'].numpy(), 
                                  parsed['state/current/velocity_y'].numpy(), 
                                  parsed['state/current/vel_yaw'].numpy(),], axis = -1)
        
        rng = np.random.default_rng()
        if np.sum(objects_of_interest == 1) < 2:
            if np.sum(tracks_to_predict == 1) < 2:
                return None
            else:
                indices = rng.choice(np.argwhere(tracks_to_predict == 1), size = 2, replace = False, p = None)
                objects_of_interest[indices] = 1
                
        origin = np.mean(agents_polylines[np.argwhere(objects_of_interest == 1).reshape(-1), 0, : 3], axis = 0) # shape of (3,)
        
        agents_polylines[agents_polylines_mask, : 3] = agents_polylines[agents_polylines_mask, : 3] - origin
        roadgraph_polylines[roadgraph_polylines_mask, : 3] = roadgraph_polylines[roadgraph_polylines_mask, : 3] - origin
        targets[targets_mask, : 3] = targets[targets_mask, :3] - origin
        
        
        return torch.tensor(agents_polylines), torch.tensor(agents_polylines_mask), \
               torch.tensor(roadgraph_polylines), torch.tensor(roadgraph_polylines_mask), \
               torch.tensor(targets), torch.tensor(targets_mask), \
               torch.tensor(tracks_to_predict), torch.tensor(objects_of_interest), \
               torch.tensor(agents_motion)
    
if __name__ == '__main__':
    val_dataset = waymo_motion_dataset(dataroot = '/datasets-2/waymo/uncompressed/tf_example/validation_interactive',
                                       scene_list = None)

    val_loader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False, collate_fn = collate)

    print(len(val_dataset))
    
    for minibatch_count, data in enumerate(tqdm(val_loader)):

        traffic_lights_polylines, traffic_lights_polylines_mask, \
        agents_polylines, agents_polylines_mask, \
        roadgraph_polylines, roadgraph_polylines_mask, \
        targets, targets_mask,\
        tracks_to_predict, objects_of_interest, \
        agent_motion = data

        if minibatch_count == 0:
            # [batch_size, num_traffic_lights num_steps, dim_traffic_lights_vec]
            print('traffic_lights_polylines.shape', traffic_lights_polylines.shape)
            # [batch_size, num_traffic_lights, num_steps]
            print('traffic_lights_polylines_mask.shape', traffic_lights_polylines_mask.shape)
            # [batch_size, num_agents, num_steps, dim_agent_vec]
            print('agents_polylines.shape', agents_polylines.shape)
            # [batch_size, num_agents, num_steps]
            print('agents_polylines_mask.shape', agents_polylines_mask.shape)
            # [batch_size, max_num_roads (varies for different batches), max_num_roads (varies for different batches), dim_roadgraph_vecs]
            print('roadgraph_polylines.shape', roadgraph_polylines.shape)
            # [batch_size, max_num_roads (varies for different batches), max_num_roads (varies for different batches)]
            print('roadgraph_polylines_mask.shape', roadgraph_polylines_mask.shape)
            # [batch_size, num_agents, num_future_steps, dim_future_vec]
            print('targets.shape', targets.shape)
            # [batch_size, num_agents, num_future_steps]
            print('targets_mask.shape', targets_mask.shape)
            
            print('agent_motion.shape', agent_motion.shape)
