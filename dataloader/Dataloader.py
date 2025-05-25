import json
import csv

from CONSTANT import *


def insert_template(question: str, question_and_choice: str, option="多选"):  # TODO 可以改为中文版本&这里改单选或多选
    s = question + "\n选项：" + question_and_choice[len(question):]
    if option == "单选":
        s = "Q：从以下的A,B,C,D中选择选项，你只能选择一个最佳选项：" + s  # 单选实现
    elif option == "多选":
        s = "Q：从以下的A,B,C,D中选择选项，你可以选择一至多个你认为合适的选项：" + s  # 多选实现
    return s


class Dataloader:
    def __init__(self, option="csv", file_name='选择题修正数据(无空选项).csv', test=False):
        if test:
            self.data_path = os.path.join(GET_PROJECT_ROOT(), "data", "eval", "test.{}".format(option))
        else:
            self.data_path = os.path.join(GET_PROJECT_ROOT(), "data", "eval", file_name)
        self.data = dict()  # ->{q:a}

        if option == "json":  # file_name='QAOC.json'
            with open(self.data_path, 'r', encoding='utf=8') as fp:
                self.data = json.load(fp=fp)  # TODO
            # print(self.data)

        elif option == "csv":  # file_name='选择题修正数据(无空选项).csv'
            with open(self.data_path, 'r', encoding='utf-8') as fp:
                reader = csv.reader(fp, delimiter=',')
                _ = reader.__next__()  # 去表头
                for row in reader:
                    _, QA, _, Q, answer = row
                    q = insert_template(list(eval(QA))[0], Q)
                    self.data[q] = answer
            # del self.data['']
            # print(self.data)

    def get_data(self):
        return self.data


if __name__ == '__main__':
    dl = Dataloader(test=True)
