import numpy as np

def label_distribution(dataset, labels):
    dist = np.zeros((dataset.num_labels, 1))
    for i in range(0, len(labels)):
        dist[labels[i], 0] += 1
    return dist

def dataset_statistics(dataset):
    train_dist = label_distribution(dataset, dataset.y_train)
    test_dist = label_distribution(dataset, dataset.y_test)
    overall = train_dist + test_dist
    
    train_ratio = train_dist * 100 / dataset.num_train_examples
    test_ratio = test_dist * 100 / dataset.num_test_examples
    overall_ratio = overall * 100 / (dataset.num_train_examples + dataset.num_test_examples)
    
    # Fiddle around to get it into table shape
    table = np.hstack((np.zeros((dataset.num_labels,1), dtype=int), 
                       train_dist, 
                       train_ratio, 
                       test_dist, 
                       test_ratio, 
                       overall, 
                       overall_ratio))
    
    table = np.vstack((np.zeros((1, 7), dtype=int), table)).tolist()
    
    human_labels = map(dataset.human_label_for, range(0, dataset.num_labels))
    
    for i, s in enumerate(human_labels):
        table[i + 1][0] = s
    
    table.sort(lambda x, y: cmp(x[1], y[1]))
    
    table[0][0] = ""
    table[0][1] = "Train"
    table[0][2] = "Train %"
    table[0][3] = "Test"
    table[0][4] = "Test %"
    table[0][5] = "Overall"
    table[0][6] = "Overall %"
    return table