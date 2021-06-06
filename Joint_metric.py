# pos_preds is (N_actors, N_modes, T, 2)
# probs is (N_modes)
# GT is (N_actors, T, 2)
def min_K_joint_ADE_metric(pos_preds, GT):
    pred_size = list(pos_preds.size())
    T = pred_size[2]
    
    GT = GT[:,None,:,:]
    
    # shape (N_actors, N_modes, T, 2)
    sq_dif = torch.square(pos_preds - GT)
    # shape (N_actors, N_modes, T)
    L2_per_timestep = torch.sqrt(torch.sum(sq_dif, 3))
    # shape (N_actors, N_modes)
    ADE_per_actor_per_mode = torch.sum(L2_per_timestep, 2) / T
    # shape (N_modes)
    ADE_per_mode = torch.sum(ADE_per_actor_per_mode, 0)
    # shape (,)
    best_mode = torch.argmin(ADE_per_mode, 0).type(torch.LongTensor)
    min_ADE = torch.index_select(ADE_per_mode, 0, best_mode)
    
    return min_ADE