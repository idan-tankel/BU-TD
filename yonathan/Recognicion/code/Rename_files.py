import os
from os import path
path = '/home/sverkip/data/BU-TD/yonathan/Recognicion/data/omniglot/results_all_omniglot'
for path_ in os.listdir(path):
    if path_ != '5R' and path_!= '5L':
        path_ = os.path.join(path,path_)
        os.rename(path_,path_+ '_right')
        print(path_)

