import torch


# pos_preds is (N_actors, N_modes, T, 2)
# probs is (N_modes)
# GT is (N_actors, T, 2)
def multi_mode_loss_L2(pos_preds, probs, GT):
    pred_size = list(pos_preds.size())
    T = pred_size[2]
    
    GT = GT[:,None,:,:]
    
    # shape (N_actors, N_modes, T, 2)
    sq_dif = torch.square(pos_preds - GT)
    # shape (N_actors, N_modes, T)
    L2_per_timestep = torch.sqrt(torch.sum(sq_dif, 3))
    # shape (N_actors, N_modes)
    ADE_per_actor_per_mode_per_ten = torch.sum(L2_per_timestep[:, :, range(0, T, 10)], 2) / T * 10
    ADE_per_actor_per_mode = torch.sum(L2_per_timestep, 2) / T
    # shape (N_modes)
    ADE_per_mode = torch.sum(ADE_per_actor_per_mode, 0)
    # shape (,)
    best_mode = torch.argmin(ADE_per_mode, 0).type(torch.LongTensor).cuda()
    min_ADE = torch.index_select(ADE_per_mode, 0, best_mode)
    min_ADE_prob = torch.index_select(probs, 0, best_mode)
    min_ADE_CrossEnt = -1*torch.log(min_ADE_prob+1e-5)
    
    return min_ADE, min_ADE_CrossEnt
