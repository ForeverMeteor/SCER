from self_consistency.SelfConsistencyChatGPT import SelfConsistencyChatGPT
from self_consistency.SelfConsistencyChatGLM import SelfConsistencyChatGLM
from dataloader.Dataloader import Dataloader
from utils import *


import argparse
from tqdm import tqdm
import sys
import time
sys.path.extend([".", "dataloader", "knowledge", "self_consistency", "utils"])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--self_consistency_rounds', default=10, type=int, help='Number of SC path. Default 10')
    argparser.add_argument('--random_counts', default=4, type=int, help='Counts of random walk. P in paper.')
    argparser.add_argument('--random_length', default=3, type=int, help='Length of a random walk path. l in paper.')
    argparser.add_argument('--llm', default="ChatGPT", type=str, help='Use what LLM')
    argparser.add_argument('--file', default=0, type=int)
    args, extra_args = argparser.parse_known_args()

    self_consistency_rounds = args.self_consistency_rounds
    P = args.random_counts
    l = args.random_length
    llm = args.llm
    file_number = args.file

    # 读数据
    print("Loading data...")
    d = Dataloader(file_name='选择题修正数据(无空选项){}.csv'.format(file_number), test=False)
    data = d.get_data()
    print("Loading data finished.")

    sc_result = []  # -> [(index,question,[sc_path1,...],correct_answer),...]
    s = None
    if llm == "ChatGPT":
        s = SelfConsistencyChatGPT(self_consistency_rounds=self_consistency_rounds)
    elif llm == "ChatGLM":
        s = SelfConsistencyChatGLM(self_consistency_rounds=self_consistency_rounds)
    print("Using {}".format(llm))

    index = 0
    print("SC started.")
    start_time = time.time()
    for question, correct_answer in tqdm(data.items()):
        # 执行SC过程
        R = s.get_inference_paths(question=question)  # -> [str,...]
        sc_result.append((index, question, R, correct_answer))
        # 逐步覆写 防崩溃
        save_sc_result(sc_result, file_num=file_number)
        index += 1
    print("SC finished.")
