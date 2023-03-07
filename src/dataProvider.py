#mnist heuristic
def build_data_provider(local_batch_size, examples_per_user, drop_last: bool = False):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data/Experiment/data/MNIST",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data/Experiment/data/MNIST", train=False, download=True, transform=transform
    )
    client_num = num_of_original_client
    # divide train dataset(non-iid)

    dict_users = mnist_noniid(train_dataset, client_num)

    index_of_head_group, time = heuristic_method(train_dataset, num_of_MNIST_label, dict_users, client_num,
                                                 num_of_head_client)
    global total_execution_time
    total_execution_time = total_execution_time + time

    # merge train dataset
    sorted_train_dataset = []
    for k in range(num_of_head_client):
        for i in range(len(index_of_head_group[k])):
            client_index = index_of_head_group[k][i]
            for j in range(len(dict_users[client_index])):
                data_index = int(dict_users[client_index][j])
                sorted_train_dataset.append(train_dataset[data_index])

    num_of_client = int(len(train_dataset) / examples_per_user)

    KL_of_each_client, avg_KL = get_KL_value(sorted_train_dataset, num_of_MNIST_label, num_of_client)

    Save_KL_Result("FL_non_IID_Heuristic_MNIST(MLP)", KL_of_each_client, avg_KL)
    # get the amount of each class

    sharder = SequentialSharder(examples_per_shard=examples_per_user)
    fl_data_loader = DataLoader(
        sorted_train_dataset,
        test_dataset,
        test_dataset,
        sharder,
        local_batch_size,
        drop_last
    )
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider