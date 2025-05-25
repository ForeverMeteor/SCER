import time
import argparse
import sys
from tqdm import tqdm

import CONSTANT

sys.path.extend([".", "dataloader", "knowledge", "self_consistency", "utils"])

from knowledge.KnowledgeBase import KnowledgeBase
from utils import *

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--random_counts', default=4, type=int, help='Counts of random walk. P in paper.')
    argparser.add_argument('--random_length', default=3, type=int, help='Length of a random walk path. l in paper.')
    argparser.add_argument('--file', default=0, type=int)
    args, extra_args = argparser.parse_known_args()

    P = args.random_counts
    l = args.random_length
    file_number = args.file

    print("Loading SC data...")
    with open("./result/chatglm/ChatGLM_SCresult_20240329_{}.json".format(file_number), "r", encoding="utf-8") as fp:
        sc_result = json.load(fp=fp)  # -> [(index,question,[sc_path1,...],correct_answer),...]
    print("Finished loading SC data...")

    answer_list = []
    correct_answer_list = []
    k = KnowledgeBase(random_length=l, P=P)

    torch.cuda.set_device(int(CONSTANT.GET_CUDA().split(":")[-1]))

    print("KB retrieve Started")
    start_time = time.time()
    for d in tqdm(sc_result):
        index = d[0]
        question = d[1]
        R = d[2][:5]  # FIXME 此处确定3,5,10
        correct_answer = d[3]

        # 执行KB过程
        k.set_R(R)
        k.retrieve_from_KB()
        answer = k.get_most_faithful_answer()
        answer_list.append(answer)
        print("Answer:", answer)
        correct_answer_list.append(correct_answer)
        print("Correct_answer:", correct_answer)
        # 逐步覆写
        save_answer_result(answer_list, correct_answer_list, file_num=file_number, length=5)  # FIXME 此处确定3,5,10
    print("KB retrieve Finished")
    end_time = time.time()

    # 统计正确率
    print("Used time: {}s".format(end_time - start_time))
    result = count_correct(file_name="Answers_{}_{}.csv".format(datetime.datetime.now().strftime('%Y%m%d'), file_number))
    print(result)
