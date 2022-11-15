import concurrent
import datetime
import logging
import os
import threading
import time
from os import listdir
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import pool
from src.server import Server

import yaml
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from os.path import isfile, join
from src.server import Server
from src.utils import launch_tensor_board

def run(serverModel:Server):
    serverModel.fit()

def loadConfig(filePath):
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
    #log_config["log_path"] = os.path.join(log_config["log_path"],str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    for config in configs:
        print(config)
        logging.info(config)
    print()
    return configs,global_config, data_config, fed_config, optim_config, init_config, model_config, log_config

if __name__ == "__main__":
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
        #if logDir == '':
            #logDir = logDir.join(os.path.abspath(log_config["log_path"]))
        #else:
            #logDir = logDir.join(','.join(os.path.abspath(log_config["log_path"])))
        # do federated learning
        # central_server.fit()

    logging.info(message)
    time.sleep(3)
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=(['./log/', "5252", '0.0.0.0'])
    ).start()
    time.sleep(3.0)
    for server in serverList:
        server.fit()
