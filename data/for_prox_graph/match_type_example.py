import pickle
import pdb
align_dict = {}
with open('matched_types.txt', 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        align_dict[line[0]] = line[1]

meta_prox_graph = 'yago_pred_prox_graph'
aligned_prox_graph = []

with open(f"{meta_prox_graph}.txt", 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        if line[0] in align_dict:
            line[0] = align_dict[line[0]]
        if line[1] in align_dict:
            line[1] = align_dict[line[1]]
        
        aligned_prox_graph.append(line)


pickle.dump(aligned_prox_graph, open(f"{meta_prox_graph}_matched.pickle", 'wb'))

