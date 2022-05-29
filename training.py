import pickle

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

import consts
import functions


def train():
    training_data = functions.load_preprocessed_data()
    init = 'k-means++'
    max_iter=300
    n_init=10
    
    kmeans = KMeans(n_clusters=consts.n_clusters, init=init ,max_iter=max_iter, n_init=n_init, random_state=None)
    print('training KMeans...')
    try:
        kmeans.fit(training_data)
        model_name = 'kmeans.pkl'
        print(f'Saving KMeans model as {model_name}')
        pickle.dump(kmeans, open(model_name, 'wb'))
    except ValueError as e:
        print(e)
        
    imgs_num = training_data.shape[0]// training_data.shape[1]
    evaluated_histogram_list =[]
    
    for i in tqdm(range(imgs_num), 'Evaluting the training histogram'):
        evaluted_class = kmeans.predict(training_data[(i*training_data.shape[1]): (i*training_data.shape[1])+training_data.shape[1]])
        
        image_histogram, _=np.histogram(evaluted_class, bins=range(consts.n_clusters+1),density=True)
        
        evaluated_histogram_list.append(image_histogram)
        
    evaluated_histogram_list = np.array(evaluated_histogram_list)
    np.save('evaluated_histogram_list.npy', evaluated_histogram_list)
    
train()
