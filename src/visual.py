import pandas as pd


# make data
def showLabelDistribution(clientList, fileName):
    result = [[0 for _ in range(10)] for _ in range(len(clientList))]
    label = [i for i in range(10)]
    for i in label:
        for j in range(len(clientList)):
            result[j][i] = 0 if clientList[j][i] is None else clientList[j][i]
    df = pd.DataFrame(clientList, columns=label)
    df.to_csv("./" + fileName + '.csv', index=False)
