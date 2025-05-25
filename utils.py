"""
    模型各项指标统计
"""
import datetime
import json

from CONSTANT import *

def save_zeroshot_result(sc_result: list, file_num=0):
    save_path = os.path.join(GET_PROJECT_ROOT(), "result", "zeroshot")
    file_name = "Zeroshot_result_{}_{}.json".format(datetime.datetime.now().strftime('%Y%m%d'), file_num)
    with open(os.path.join(save_path, file_name), 'w', encoding="utf-8") as fp:
        json.dump(sc_result, fp, indent=4, ensure_ascii=False)


def save_sc_result(sc_result: list, file_num=0):
    save_path = os.path.join(GET_PROJECT_ROOT(), "result", "chatglm")
    file_name = "ChatGLM_SCresult_{}_{}.json".format(datetime.datetime.now().strftime('%Y%m%d'), file_num)  # FIXME
    with open(os.path.join(save_path, file_name), 'w', encoding="utf-8") as fp:
        json.dump(sc_result, fp, indent=4, ensure_ascii=False)


def save_answer_result(answers: list, correct_answers: list, file_num=0, length=-1):  # -> [AB,....]; -> [ABD,...]
    save_path = os.path.join(GET_PROJECT_ROOT(), "result/chatglm", "length{}".format(length))
    file_name = "Answers_{}_{}.csv".format(datetime.datetime.now().strftime('%Y%m%d'), file_num)
    with open(os.path.join(save_path, file_name), 'w', encoding="utf-8") as fp:
        for answer, correct_answer in zip(answers, correct_answers):
            fp.write("{},{}\n".format(answer, correct_answer))


def count_correct(file_name):
    save_path = os.path.join(GET_PROJECT_ROOT(), "result")

    total_point = 0
    point_sum = 0
    with open(os.path.join(save_path, file_name), 'r', encoding="utf-8") as fp:
        for line in fp:
            total_point += 1
            try:
                answer, correct_answer = line.split(',')
            except ValueError:
                continue

            correct_cnt = 0
            for choice in answer.strip():
                if choice in correct_answer:
                    correct_cnt += 1
                else:
                    correct_cnt = 0
                    break
            # print(correct_cnt, len(correct_answer.strip()))
            point_sum += correct_cnt / len(correct_answer.strip())

    return "Point: %.4f%%" % (100 * point_sum / total_point)


if __name__ == '__main__':
    file_number = 5
    result = count_correct(file_name="Answers_{}_{}.csv".format(datetime.datetime.now().strftime('%Y%m%d'), file_number))
    print(result)
