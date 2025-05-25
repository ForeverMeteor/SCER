import re

import numpy as np
import sys

import torch
from sentence_transformers.util import cos_sim

import CONSTANT

sys.path.extend([".", ".."])

from Graph import Graph
from Separator import Separator
from NER import NER
from M3E import M3E
from NLImodel import NLI


'''
    论文中的Dtime函数和concat函数结合体
    :return res -> list  [node,...]
    :author ChentaoZhang 2023.3.18
'''
def D_time_and_concat(seqs):  # seqs:[[node1, node2,...],...]
    s = set()
    res = []
    for lst in seqs:
        for node in lst:
            if node not in s:
                res.append(node)
                s.add(node)
    return res
'''
    改造自RR中的check_conclusion_sentence，用以判断结论句
'''
# def check_conclusion_sentence(sent):
#     sent = sent.replace('\n', '').strip().lower()
#     if sent.startswith('thus') or sent.startswith('therefore') or sent.startswith('so ') or sent.startswith('no,') \
#             or sent.startswith('yes,') or sent.startswith('no.') or sent.startswith('yes.') or sent == 'no' or sent == 'yes':
#         return True
#     return False


'''
    实现忠实分数计算，可更改实现；论文提供了若干种不同的忠实分数公式，这里是最简单的一种
    :return res -> np.float64
    :author ChentaoZhang 2023.3.18
'''
def faithful_func(f_KB_point_argument_list):  # f_KB_point_argument_list -> [(M_ei, E_ei, C_ei),...]
    similarity_threshold = 0.5
    res = np.float64(0)
    for M_ei, E_ei, C_ei in f_KB_point_argument_list:
        # res += M_ei + E_ei  # 实现方法1

        # 实现方法2
        if M_ei >= similarity_threshold:
            res += M_ei
        else:
            res += E_ei
        res -= C_ei

    return res / len(f_KB_point_argument_list)


def get_real_answer(s):  # s -> str
    # print("Pi:", s)
    answer_list = ''
    if 'A' in s:
        answer_list += 'A'
    if 'B' in s:
        answer_list += 'B'
    if 'C' in s:
        answer_list += 'C'
    if 'D' in s:
        answer_list += 'D'
    if 'E' in s:
        answer_list += 'E'
    if 'F' in s:
        answer_list += 'F'
    if 'G' in s:
        answer_list += 'G'
    return answer_list  # -> "AB..."


class KnowledgeBase:
    def __init__(self, random_length, P):
        self.R = None
        self.graph = Graph()
        self.random_length = random_length
        self.sentence_separator = Separator(option='by_model')
        self.ner = NER()
        self.P = P
        self.m3e = M3E()
        self.nli = NLI()
        self.top_k = 1
        self.f_KB_point_list = []
        self.answer_list = []
        torch.cuda.set_device(int(CONSTANT.GET_CUDA().split(":")[-1]))

    def set_R(self, R):
        self.R = R

    def sentence_cos(self, sentence, evidence):  # sentence -> str, evidence -> str
        v1 = self.m3e.encode(sentence)
        v2 = self.m3e.encode(evidence)
        return cos_sim(v1, v2)

    def entailment_score(self, sentence, evidences):  # sentence -> str, evidence -> [[random_walk, cos],...]
        predicted_probability_list = self.nli.get_entailment_scores(sentence, [x[0] for x in evidences])
        res = []
        for i, (random_walk, _cos) in enumerate(evidences):
            res.append((random_walk, _cos, predicted_probability_list[i]))
        return res  # ->[(random_walk, cos, (softmax)(蕴含分数,中性分数,矛盾分数)),...]

    '''
        KB查询过程，总函数
        :return seq -> list  [node,...]
        :author ChentaoZhang 2023.3.18
        
    '''
    def retrieve_from_KB(self):
        # 清空记录
        # 这里没设计好
        self.f_KB_point_list = []
        self.answer_list = []

        # 对于每一条SC做实体识别
        for Ei in self.R:
            Pi = None
            f_KB_point_argument_list = []  # -> [(M_ei, E_ei, C_ei),...]

            # 分句
            Ei = re.sub("\n+", "", Ei)
            e = self.sentence_separator.separate(Ei)
            # 对每一小句做实体识别
            for index, ei in enumerate(e):
                if index == len(e)-1:  # 结论句
                    Pi = ei

                # NER
                entities_i = None
                for i in range(0, 12):  # 重复试验12次
                    entities_i = self.ner.gpt_ner(ei)  # [entity1,...]
                    # print(entities_i)
                    entities_i = self.ner.trans_into_list(entities_i)
                    if entities_i is not None:
                        break
                if entities_i is None:
                    entities_i = []

                # 对于每一小句做句子级别的随机游走，重复10次
                KG_ei = []
                for i in range(0, self.P):
                    # 对于每一实体做实体级别的随机游走，长度为3
                    seqs = []
                    for entity in entities_i:
                        node = self.graph.get_node_through_entitystr(entity)
                        if node is not None:
                            # random_walk() return seq -> list[node, ...]
                            seqs.append(self.graph.random_walk(node, self.random_length))
                        else:
                            # print("Entity \"{}\" not in Graph".format(entity))
                            pass
                    # print("seqs:", seqs)
                    # end for entity

                    # 按时序拼接，KG_ei_p -> [node's index,...]
                    KG_ei.append(D_time_and_concat(seqs))
                # print("KG_ei:", KG_ei)
                # end for P

                # 计算依据夹角列表，即cos和entailment
                evidence_list = []
                assert len(KG_ei) == self.P
                for p in range(0, len(KG_ei)):
                    # KG_ei_p = KG_ei[p]
                    # 把node编号转化为含解释的句子
                    KG_ei_p = []
                    for x in KG_ei[p]:
                        entity, description = self.graph.get_node_through_id(x)
                        KG_ei_p.append("{}，术语描述：{}".format(entity, description))
                    KG_ei_p = ' '.join(KG_ei_p)

                    # 计算cos夹角
                    _cos = np.float64(self.sentence_cos(ei, KG_ei_p))
                    # print("_cos:", _cos)
                    evidence_list.append((KG_ei_p, _cos))
                evidence_list.sort(key=lambda a: a[1], reverse=True)

                # 计算NLI得分（因batch所需）
                evidence_list = self.entailment_score(ei, evidence_list)
                # print("evidence_list[0]:", evidence_list[0])  # ->[(random_walk, cos, (softmax)(蕴含分数,中性分数,矛盾分数)),...]

                # 计算faithful function的参数
                M_ei = -2.0
                E_ei = 0.0
                C_ei = 0.0
                for evidence in evidence_list[:self.top_k]:  # top_k=1
                    # 统计从所有的各个方面来看，这个分句的可信程度怎么样，所以将一个分句的所有的依据取最高值
                    # evidence -> [random_walk, cos, (蕴含分数,？,矛盾分数)]
                    # print("evidence:", evidence)
                    if evidence[1] > M_ei:  # cos
                        M_ei = evidence[1]
                    if evidence[2][0] > E_ei:
                        E_ei = evidence[2][0]
                    if evidence[2][2] > C_ei:
                        C_ei = evidence[2][2]

                f_KB_point_argument_list.append((M_ei, E_ei, C_ei))
            # end for ei

            # 计算faithful function
            # print("f_KB_point_argument_list:", f_KB_point_argument_list)
            f_KB_point = faithful_func(f_KB_point_argument_list)
            self.f_KB_point_list.append(f_KB_point)

            # 获得这一Ri的答案（可能是多选）
            answers = get_real_answer(Ei)
            self.answer_list.append(answers)

        print("f_KB_point_list:", self.f_KB_point_list)
        print("answer_list:", self.answer_list)
        # end for R

        return
    # end def

    def get_most_faithful_answer(self):
        try:
            assert len(self.f_KB_point_list) == len(self.R)
            assert len(self.answer_list) == len(self.R)
        except AssertionError as e:
            print("self.f_KB_point_list: {}".format(self.f_KB_point_list))
            print("self.answer_list: {}".format(self.answer_list))
            print("self.R: {}".format(self.R))
            raise e

        # 全一样
        if len(set(self.answer_list)) == 1:
            return self.answer_list[0]

        d = dict()  # d:{choice:point}
        for faithful_point, answers in zip(self.f_KB_point_list, self.answer_list):
            for answer in answers:
                if answer not in d:
                    d[answer] = np.float64(0)
                d[answer] += faithful_point

        # 得出最大值做结果，这里是单选题的实现
        # max_point = -2
        # most_faithful_answer = None
        # for answer, point in d.items():
        #     if point >= max_point:
        #         max_point = point
        #         most_faithful_answer = answer

        # 多选题实现
        choice_threshold = 0.45  # FIXME 修改阈值
        most_faithful_answer = ""
        for choice, point in d.items():
            if point / len(self.R) > choice_threshold:
                most_faithful_answer += choice
        if len(most_faithful_answer) == 0:
            print("Answer empty")
            # 做单选，选最高的
            max_point = -2
            for answer, point in d.items():
                if point > max_point:
                    max_point = point
                    most_faithful_answer = answer

        # print("most_faithful_answer:", most_faithful_answer)
        return most_faithful_answer  # -> str


if __name__ == '__main__':
    k = KnowledgeBase(random_length=2,
                      P=1)
    k.set_R(["I am fool. So the answer is A.", "I am not fool. So the answer is B."])
    k.retrieve_from_KB()
    k.get_most_faithful_answer()
