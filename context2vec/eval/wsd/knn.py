'''
K-nearest neighbor classifier
'''
import numpy as np

class Knn(object):

    def __init__(self, k, dataset, key2ind):
        self.k = k
        self.dataset = dataset
        self.key2ind = key2ind
        
    def classify(self, key, vec, ignore_closest, debug):
        
        norm = np.sqrt(vec.dot(vec))
        vec /= norm
        
        key_data = self.dataset[self.key2ind[key]]
        similarity = key_data.context_m.dot(vec)
        
        result = {}
        neighbors_found = 0
        for rank, ind in enumerate((-similarity).argsort()):
            if ignore_closest and rank==0:
                continue
            text = key_data.contexts_str[ind]
            sense_ids = key_data.sense_ids[ind]
            weight = 1./len(sense_ids)
            if debug:
                print('RET: ', similarity[ind], text.strip(), sense_ids)
            for sid in sense_ids:
                if sid not in result:
                    result[sid] = weight
                else:
                    result[sid] += weight
            neighbors_found += 1
            if neighbors_found == self.k:
                break
        return result
        
        
        