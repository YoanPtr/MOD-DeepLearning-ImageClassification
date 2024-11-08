import numpy as np
import os 

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar_batch(path):
    batch = unpickle(path)

    data = batch[b'data']
    label = batch[b'labels']
    return data,label 


def read_cifar(path): 
    filenames = ['data_batch_' + str(i) for i in range(1,6)] + ['test_batch']

    data = None
    labels= None

    for filename in filenames:
        filepath = os.path.join(path,filename)
        data_batch,label = read_cifar_batch(filepath)
        if data is None : 
            data = data_batch
            labels = label
        else : 
            data = np.concatenate((data,data_batch), axis=0)
            labels = np.concatenate((labels,label), axis=0)

    return data,labels

def read_cifar_test(path):

    filepath = os.path.join(path,'test_batch')
    data,labels = read_cifar_batch(filepath)

    labels = np.array(labels)
    
    return data,labels

def split_dataset(data,labels,split): 
    length = data.shape[0]
    n = int(length*split)
    indices = np.arange(length)

    train_indices = np.random.choice(indices, size=n, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    
    data_train = data[train_indices]
    labels_train = labels[train_indices]

    data_test = data[test_indices]
    labels_test = labels[test_indices]

    return data_train,labels_train,data_test,labels_test

def normalized(data):
    return (data / 255.0 ) - 0.5

def unnormalized(data):
    return 2 * (data * 255.0) 

def get_label(path):
    filepath = os.path.join(path, 'batches.meta')

    meta = unpickle(filepath)
    label_names = meta[b'label_names']  # A list of 10 label names
    label_names = [label.decode('utf-8') for label in label_names]
    return label_names

if __name__ == "__main__":


    #path = 'data/cifar-10-batches-py/test_batch'
    #data,label  = read_cifar_batch(path)

    path = 'data/cifar-10-batches-py/'
    #data,labels  = read_cifar(path)
    labels = get_label(path)
    data_train,labels_train,data_test,labels_test = split_dataset(data,labels,0.7)



