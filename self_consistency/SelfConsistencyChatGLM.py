import time
from CONSTANT import *

from transformers import AutoTokenizer, AutoModel
from SelfConsistency import SelfConsistency


class SelfConsistencyChatGLM(SelfConsistency):
    def __init__(self, self_consistency_rounds):
        super().__init__(self_consistency_rounds)

        start_time0 = time.time()
        print("Start loading ChatGLM3-6b")
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).half().cuda()
        self.model = model.eval()
        end_time0 = time.time()
        print("Load Time: {}".format(end_time0 - start_time0))

        with open(os.path.join(GET_PROJECT_ROOT(), "data", "prompt", "CoT.txt"), 'r', encoding='utf-8') as fp:
            s = ""
            for line in fp:
                if line.startswith('#'):
                    continue
                s += line.strip() + '\n'
            self.CoT = s

    def get_inference_paths(self, question, test=False):
        paths = []
        print("Getting response of LLM, total round {}".format(self.self_consistency_rounds))
        if test:
            for i in range(self.self_consistency_rounds):
                print("Round {}".format(i + 1))
                paths.append(
                    "Yes. The official language of Saint Vincent and the Grenadines is English. So the answer is yes.")
                print("Round {} finished".format(i + 1))

        else:
            for i in range(self.self_consistency_rounds):
                res, _ = self.model.chat(self.tokenizer, question, history=[])
                paths.append(res)
                # print("Round {} finished".format(i + 1))

        print("LLM Finished responding.")
        return paths  # -> [str,...]


if __name__ == '__main__':
    sc = SelfConsistencyChatGLM(5)

    with open(os.path.join(GET_PROJECT_ROOT(), "data", "prompt", "CoT.txt"), 'r', encoding='utf-8') as fp:
        s = ""
        for line in fp:
            if line.startswith('#'):
                continue
            s += line.strip() + '\n'
        CoT = s
    q = CoT + "Q:从A,B,C,D中选择一至多个合适的选项：电力电缆线路应进行什么测量？A: 电力电缆线路接地电阻测量 B: 电力电缆线路局部放电测量 C: 断路器在线监测 D: 低压故障穿越监测\nA: "

    start_time = time.time()
    p = sc.get_inference_paths(question=q, test=False)
    print(p)
    end_time = time.time()
    print("Time used: {}".format(end_time - start_time))
