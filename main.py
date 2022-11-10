<<<<<<< HEAD
import concurrent
import datetime
import logging
import os
import threading
import time
from os import listdir
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor
=======
import datetime
import logging
import os
import pickle
import threading
import time
>>>>>>> e76e0df31813d23a43c7bbf01abbe4fa56f16814

import yaml
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile, join
from src.server import Server
from src.utils import launch_tensor_board


<<<<<<< HEAD
def loadConfig(filePath):
=======
def readConfigFile(filePath):
    # read configuration file
>>>>>>> e76e0df31813d23a43c7bbf01abbe4fa56f16814
    with open(filePath) as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    optim_config = configs[3]["optim_config"]
    init_config = configs[4]["init_config"]
    model_config = configs[5]["model_config"]
    log_config = configs[6]["log_config"]
    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(log_config["log_path"],
                                          str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    for config in configs:
        print(config)
        logging.info(config)
    print()
    return global_config, data_config, fed_config, optim_config, init_config, model_config, log_config

<<<<<<< HEAD
    return configs, global_config, data_config, fed_config, optim_config, init_config, model_config, log_config


if __name__ == "__main__":
    # initiate TensorBaord for tracking losses and metrics
    serverList = []
    mypath = "./configs/"
    logDir = ""
    processList = []
    for fileName in [fileName for fileName in listdir(mypath) if isfile(join(mypath, fileName))]:
        configs, global_config, data_config, fed_config, optim_config, init_config, model_config, log_config = loadConfig(
            mypath + fileName)

        # display and log experiment configuration
        message = "\n[WELCOME] Unfolding configurations...!"
        print(message)
        logging.info(message)

        # set the configuration of global logger
        logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename=os.path.join(log_config["log_path"], log_config["log_name"]),
            level=logging.INFO,
            format="[%(levelname)s](%(asctime)s) %(message)s",
            datefmt="%Y/%m/%d/ %I:%M:%S %p")

        for config in configs:
            print(config)
            logging.info(config)
        print()
        writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
        # initialize federated learning
        central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config,
                                log_config)
        central_server.setup()
        serverList.append(central_server)
        if logDir == '':
            logDir = logDir.join(os.path.abspath(log_config["log_path"]))
        else:
            logDir = logDir.join(','.join(os.path.abspath(log_config["log_path"])))
        # do federated learning
        # central_server.fit()

    logging.info(message)
    time.sleep(3)
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=(['./logs/', "5252", '0.0.0.0'])
    ).start()
    time.sleep(3.0)
    with ThreadPoolExecutor() as executor:
        result = [executor.submit(target=server.fit()) for server in serverList]
        for f in concurrent.futures.as_completed(result):
            print(f.result())

=======

if __name__ == "__main__":
    _, _, fed_config, _, _, _, log_config = readConfigFile("./config.yaml")

    writer = SummaryWriter(log_dir=log_config["log_path"],
                           filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"],
               log_config["tb_port"],
               log_config["tb_host"]])
    ).start()
    time.sleep(3.0)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"],
                              log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message)
    logging.info(message)

    configFileList = [f for f in listdir('./configs') if isfile(join('./configs', f))]
    for file in configFileList:
        global_config, data_config, fed_config, optim_config, init_config, model_config, log_config = readConfigFile("./configs/"+file)
        # initialize federated learning
        central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config, optim_config)
        central_server.setup()
        # do federated learning
        central_server.fit()
        # save resulting losses and metrics
        with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
            pickle.dump(central_server.results, f)
>>>>>>> e76e0df31813d23a43c7bbf01abbe4fa56f16814

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message)
