# VecterNet-based-motion-prediction.\
This is a VecterNet based motion prediction for Waymo interactive prediction challenge\
Graph output is of size (#agents + #roadmap_features)*64\
Agents of interest index can be obtained using tf.where(parsed['state/objects_of_interest'] == 1).numpy()\
So nodes nos corresponding to  above in the graph correspond to nodes of the agents of interest.\
Currently, traffic light features are not included in the working. Fututre work will focus on including traffic light features into the graph interaction. \
Agent polyline will be of shape #agents *10 *5 i.e. for each agent, we have 10 vectors each having 5 features, the starting and ending position, timestamp, type and id. \
Roadmap polyline will be of shape #roadmap_features * max_length of vector in the roadmap * 8. Entries beyond the size of current roadmap size are filled with large negative numbers so that they dont have an effect on the graph because of maxpooling.\
