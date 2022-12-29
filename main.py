import concurrent
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from os import listdir
from os.path import isfile, join
from pathlib import Path

import yaml
from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board


def loadConfig(filePath):
    # read configuration fil
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
    # log_config["log_path"] = os.path.join(log_config["log_path"],str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    for config in configs:
        print(config)
        logging.info(config)
    print()
    return configs, global_config, data_config, fed_config, optim_config, init_config, model_config, log_config


def loadConfigDir(path):
    result = []
    for fileName in listdir(path):
        subPath = path+"/"+fileName
        subPathDir = Path(subPath)
        if isfile(subPath) and fileName.endswith(".yaml"):
            result.append(subPath)
        if subPathDir.is_dir():
            result.extend(loadConfigDir(subPath))
    return result


if __name__ == "__main__":
    serverList = []
    mypath = "./configs"
    logDir = ""
    processList = []
    filePathList = loadConfigDir(path=mypath)
    for path in filePathList:
        configs, global_config, data_config, fed_config, optim_config, init_config, model_config, log_config = loadConfig(path)

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
        # if logDir == '':
        # logDir = logDir.join(os.path.abspath(log_config["log_path"]))
        # else:
        # logDir = logDir.join(','.join(os.path.abspath(log_config["log_path"])))
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
    with ThreadPoolExecutor() as executor:
        result = [executor.submit(target=server.fit()) for server in serverList]
        for f in concurrent.futures.as_completed(result):
            print(f.result())
