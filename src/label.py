from collections import Counter


def label_group(
        sorted_train_dataset,
        groupSize):
    totalData = len(sorted_train_dataset)
    data_per_group = int(len(sorted_train_dataset) / groupSize)
    label_per_group = []
    labelcount = []
    for subset in range(0, totalData, data_per_group):
        subData = sorted_train_dataset[subset:subset + data_per_group]
        labellist = [data[1] for data in subData]
        #print("labellist:",labellist)
        map = Counter(labellist)
        #print("map:",map)
        sum = 0
        for _ in map.values():
            sum += 1
        label_per_group.append(sum)
        labelcount.append(map)
    return label_per_group, labelcount
