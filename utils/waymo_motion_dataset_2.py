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
    traffic_lights_polylines, traffic_lights_polylines_mask, \
    agents_polylines, agents_polylines_mask, \
    roadgraph_polylines, roadgraph_polylines_mask, \
    targets, targets_mask, \
    tracks_to_predict, objects_of_interest = map(list, zip(*samples))
    
    batch_size = len(roadgraph_polylines)
    max_num_roads = np.max([polyline.shape[0] for polyline in roadgraph_polylines])
    max_road_length = np.max([polyline.shape[1] for polyline in roadgraph_polylines])
    dim_roadgraph_vec = roadgraph_polylines[0].shape[2]

    roadgraph_polylines_padded = torch.zeros(batch_size, max_num_roads, max_road_length, dim_roadgraph_vec)
    roadgraph_polylines_padded.new_tensor(-1e10)
    roadgraph_polylines_mask_padded = torch.zeros(batch_size, max_num_roads, max_road_length)
    for batch_ind in range(batch_size):
        num_roads, road_length, _ = roadgraph_polylines[batch_ind].shape
        roadgraph_polylines_padded[batch_ind, : num_roads, : road_length, :] = roadgraph_polylines[batch_ind]
        roadgraph_polylines_mask_padded[batch_ind, : num_roads, : road_length] = roadgraph_polylines_mask[batch_ind]
    
    return torch.stack(traffic_lights_polylines), torch.stack(traffic_lights_polylines_mask), \
           torch.stack(agents_polylines), torch.stack(agents_polylines_mask), \
           roadgraph_polylines_padded, roadgraph_polylines_mask_padded, \
           torch.stack(targets), torch.stack(targets_mask), \
           torch.stack(tracks_to_predict), torch.stack(objects_of_interest)

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
            self.scene_list = range(self.num_scenes)
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
        
        self.data = [self.extract_data(ind) for ind in tqdm(self.scene_list) if self.extract_data(ind)]
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
        
        
        tracks_to_predict = parsed['state/tracks_to_predict'].numpy()
        objects_of_interest = parsed['state/objects_of_interest'].numpy()
        
        if np.sum(objects_of_interest==1)<2: 
            return None
        else:
            
            # process traffice light data
            
            _, num_traffic_lights = parsed['traffic_light_state/past/state'].shape

            past_traffic_lights = np.dstack([parsed['traffic_light_state/past/x'],
                                             parsed['traffic_light_state/past/y'],
                                             parsed['traffic_light_state/past/z'], 
                                             np.tile(parsed['traffic_light_state/past/timestamp_micros'].numpy().reshape(-1, 1), 
                                                     (1, num_traffic_lights)), 
                                             parsed['traffic_light_state/past/state'], 
                                             parsed['traffic_light_state/past/id']])
            current_traffic_lights = np.dstack([parsed['traffic_light_state/current/x'],
                                                parsed['traffic_light_state/current/y'],
                                                parsed['traffic_light_state/current/z'], 
                                                np.tile(parsed['traffic_light_state/current/timestamp_micros'].numpy().reshape(-1, 1), 
                                                        (1, num_traffic_lights)), 
                                                parsed['traffic_light_state/current/state'], 
                                                parsed['traffic_light_state/current/id']])


            traffic_lights_polylines = np.concatenate([current_traffic_lights, past_traffic_lights], axis = 0)
            traffic_lights_polylines_mask = np.concatenate([parsed['traffic_light_state/current/valid'],
                                                            parsed['traffic_light_state/past/valid']], axis = 0).astype(np.bool)
            traffic_lights_polylines = np.swapaxes(traffic_lights_polylines, 0, 1)
            traffic_lights_polylines_mask = np.swapaxes(traffic_lights_polylines_mask, 0, 1)
        #         print('traffic_lights_polylines.shape', traffic_lights_polylines.shape)
        #         print('traffic_lights_polylines_masks.shape', traffic_lights_polylines_mask.shape)
        #         print('traffic_lights_polylines[traffic_lights_polylines_mask]', traffic_lights_polylines[traffic_lights_polylines_mask, :].shape)


            # process past and current agent data        
            num_agents, num_past_steps = parsed['state/past/x'].shape
            _, num_current_steps = parsed['state/current/x'].shape
            past_agents = np.dstack([parsed['state/past/x'].numpy(), 
                                     parsed['state/past/y'].numpy(), 
                                     parsed['state/past/z'].numpy(),
                                     np.tile(parsed['state/type'].numpy().reshape(-1, 1), (1, num_past_steps)),
                                     np.tile(parsed['state/id'].numpy().reshape(-1, 1), (1, num_past_steps))])
            current_agents = np.dstack([parsed['state/current/x'].numpy(), 
                                     parsed['state/current/y'].numpy(), 
                                     parsed['state/current/z'].numpy(),
                                     np.tile(parsed['state/type'].numpy().reshape(-1, 1), (1, num_current_steps)),
                                     np.tile(parsed['state/id'].numpy().reshape(-1, 1), (1, num_current_steps))])



            agents_polylines = np.concatenate([current_agents, past_agents], axis = 1)
            agents_polylines_mask = np.concatenate([parsed['state/current/valid'],
                                                    parsed['state/past/valid']], axis = 1).astype(np.bool)

            # process roadmap data
            roadgraph_id_set = set([ind for ind in parsed['roadgraph_samples/id'].numpy().reshape(-1).tolist() if ind > 0])
            roadgraph_vecs = np.concatenate([parsed['roadgraph_samples/xyz'],
                                             parsed['roadgraph_samples/dir'],
                                             parsed['roadgraph_samples/type'],
                                             parsed['roadgraph_samples/id']], axis = 1)
            roadgraph_vecs_mask = parsed['roadgraph_samples/valid'].numpy().astype(np.bool).reshape(-1)

            num_points, dim_roadgraph_vecs = roadgraph_vecs.shape

            num_roads = len(roadgraph_id_set)
            max_road_length = np.max([np.sum(roadgraph_vecs[:, 7] == ind) for ind in roadgraph_id_set])

            roadgraph_polylines = np.zeros(shape=(num_roads, max_road_length, dim_roadgraph_vecs))
            roadgraph_polylines_mask = np.zeros(shape=(num_roads, max_road_length), dtype=np.bool)

            for i, ind in enumerate(roadgraph_id_set):
                indices = roadgraph_vecs[:, 7] == ind
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

            
            '''
            rng = np.random.default_rng()
            if np.sum(objects_of_interest == 1) == 0:
                indices = rng.choice(np.argwhere(parsed['state/current/valid']==1), size = 1, replace = False, p = None)
                objects_of_interest[indices] = 1
            elif np.sum(objects_of_interest==1)==1:
                indices = rng.choice(np.argwhere(parsed['state/current/valid']==1), size = 0, replace = False, p = None)
                objects_of_interest[indices] = 1
            else: 
                pass
            '''
            
            # change origin

            origin = np.mean(agents_polylines[np.argwhere(objects_of_interest == 1).reshape(-1), 0, : 3], axis = 0) # shape of (3,)
            traffic_lights_polylines[traffic_lights_polylines_mask, : 3] = traffic_lights_polylines[traffic_lights_polylines_mask, : 3] - origin
            agents_polylines[agents_polylines_mask, : 3] = agents_polylines[agents_polylines_mask, : 3] - origin
            roadgraph_polylines[roadgraph_polylines_mask, : 3] = roadgraph_polylines[roadgraph_polylines_mask, : 3] - origin
            targets[targets_mask, : 3] = targets[targets_mask, :3] - origin


            return torch.tensor(traffic_lights_polylines), torch.tensor(traffic_lights_polylines_mask), \
                   torch.tensor(agents_polylines), torch.tensor(agents_polylines_mask), \
                   torch.tensor(roadgraph_polylines), torch.tensor(roadgraph_polylines_mask), \
                   torch.tensor(targets), torch.tensor(targets_mask), \
                   torch.tensor(tracks_to_predict), torch.tensor(objects_of_interest)
    
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
        tracks_to_predict, objects_of_interest = data

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
            
            print(agents_polylines[agents_polylines_mask, :])
