import os
import sys
import torch

sys.path.extend([".", ".."])

# FIXME 此处修改部署在何处
where_cuda = "cuda:6" if torch.cuda.is_available() else "cpu"

# FIXME 此处修改镜像网站
url = "YOUR_OWN_URL"

# FIXME 此处修改API_KEY
api_key = "YOUR_OWN_KEY"


def GET_PROJECT_ROOT():
    current_abspath = os.path.abspath('__file__')
    while True:
        if os.path.split(current_abspath)[1] == 'LLMCooperate':
            project_root = current_abspath
            break
        else:
            current_abspath = os.path.dirname(current_abspath)
    return project_root


def GET_CUDA():
    return where_cuda


def GET_URL():
    return url


def GET_KEY():
    return api_key


if __name__ == '__main__':
    GET_PROJECT_ROOT()
    GET_CUDA()
