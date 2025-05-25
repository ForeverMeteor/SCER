import os
import random

import numpy as np
import openpyxl

import CONSTANT


def clean_description(description):
    import re
    description = re.sub('[\r\n]+', '', description)
    description = re.sub('\\s+', '', description)
    description = re.sub('\"class=.*>', '', description)
    description = re.sub('[?].+$', '', description)
    description = re.sub('^术语描述：', '', description)
    return description

def generate_list(length):
    return sorted([random.random() for _ in range(0, length)])


class Graph:
    def __init__(self):
        self.graph_root = os.path.join(CONSTANT.GET_PROJECT_ROOT(), "data", "graph")
        self.adjacency = dict()  # {node:{(node', probability),...}}，node就是node编号，全局统一
        self.node_info = []  # [(entity, description),...]，下标就是node编号，全局统一
        self.entity_str2node = dict()

        # read in Graph 读入图
        with open(os.path.join(self.graph_root, "edge_data.csv"), 'r', encoding='utf-8') as fp:
            for line in fp:
                entity_list = line.split(',')
                try:
                    node1 = int(entity_list[0].strip())
                    node2 = int(entity_list[1].strip())
                    p = np.float64(entity_list[2])  # FIXME 数据类型出问题可以修改这里

                    if node1 not in self.adjacency.keys():
                        self.adjacency[node1] = set()
                    self.adjacency[node1].add((node2, p))

                except ValueError as e:
                    if not e.args[0].endswith('\'e1_idx\''):
                        print(e.args[0])
                        continue
        # end with

        # read in Node information 读入节点信息
        wb = openpyxl.load_workbook(os.path.join(self.graph_root, "图节点总数据.xlsx"))
        sheet = wb["图节点总数据"]
        entity_list = list(sheet.values)[1:]  # [(entity, description),...]
        for i, (entity, description) in enumerate(entity_list):
            self.node_info.append((entity.strip(), clean_description(description)))
            self.entity_str2node[entity.strip()] = i
    # end def

    '''
        根据实体字符串获取实体的node_id
       :return node -> int / None(不存在该实体)
       :author ChentaoZhang 2023.3.21
    '''
    def get_node_through_entitystr(self, s):
        if s.strip() in self.entity_str2node.keys():
            return self.entity_str2node[s.strip()]
        else:
            # FIXME 可用cos最接近匹配
            return None

    '''
        根据node_id获取实体和解释
        :return entity, description
        :author ChentaoZhang 2023.3.21
    '''
    def get_node_through_id(self, id):
        entity, description = self.node_info[id]
        return entity, description

    '''
       获取随机游走路径中某个节点的下一节点
       :return node -> int
       :author ChentaoZhang 2023.3.18
    '''
    def get_next(self, now_node):
        try:
            assert isinstance(now_node, int)
            if now_node not in self.adjacency.keys():
                raise KeyError
        except AssertionError:
            print("Node not a int:", now_node)
            return None
        except KeyError:
            print("Node have no neighbour:", now_node)
            return None

        node_list = []  # 节点
        cumulative_probability_list = []  # 累计概率
        sum_now = 0.0

        # 1.概率归一化 FIXME 可改为非归一化概率的不跳转版本
        l = list(self.adjacency[now_node])
        random.shuffle(l)
        p_sum = 0.0
        for _, probability in l:
            p_sum += probability
        for node, probability in l:
            node_list.append(node)
            cumulative_probability_list.append(sum_now + probability/p_sum)
            sum_now += probability/p_sum
        assert abs(1-sum_now) <= 1e-6

        # 2.轮盘赌算法选择下一步游走到的点
        random_list = generate_list(len(node_list))
        assert len(random_list) == len(cumulative_probability_list)
        # print(random_list)
        for i in range(0, len(random_list)):
            if cumulative_probability_list[i] >= random_list[i]:
                return node_list[i]
            else:
                continue
        return len(node_list) - 1

    # end def

    '''
        获取一条随机游走的路径，路径长为超参数L
        :return seq -> list  [node,...]
        :author ChentaoZhang 2023.3.18
    '''
    # TODO 加入非随机跳转
    def random_walk(self, start_node, path_length):
        seq = [start_node]
        new_node = start_node
        for i in range(0, path_length - 1):
            new_node = self.get_next(new_node)
            if new_node is None:
                return seq
            seq.append(new_node)
        return seq

    '''
        将随机路径格式化输出，稍加修改即可变为prompt的一部分
        :return seq -> list  [node,...]
        :author ChentaoZhang 2023.3.18
    '''
    def format_walk_sequence(self, seq):
        print("随机路径如下：")
        for node in seq:
            print("{}，实体为：{}，解释为：{}".format(node, self.node_info[node][0], self.node_info[node][1]))
    # end def
# end class


if __name__ == '__main__':
    graph = Graph()
    seq = graph.random_walk(6501, 2)
    graph.format_walk_sequence(seq)

