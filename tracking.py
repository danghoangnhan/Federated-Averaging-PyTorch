import threading
from glob import glob

from src.utils import launch_tensor_board

lists = glob("./log/*", recursive=False)
sample = ''
for i in range(len(lists)):
    sample += "iid_"+str(i)+":"+lists[i]
    if i != len(lists) - 1:
        sample += ','
print(sample)
tb_thread = threading.Thread(target=launch_tensor_board, args=([sample, "5252", '0.0.0.0'])).start()
