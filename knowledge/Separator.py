import torch

import CONSTANT


class Separator:
    def __init__(self, option='by_model'):
        torch.cuda.set_device(int(CONSTANT.GET_CUDA().split(":")[-1]))
        self.option = option
        if option == 'by_model':
            import spacy
            self.nlp = spacy.load("zh_core_web_md")
        elif option == 'by_punctuations':
            self.nlp = None

    def separate(self, sentence):
        res = []
        if self.option == 'by_model':
            sp = self.nlp(sentence).sents
            for span in sp:
                res.append(str(span))
        elif self.option == 'by_punctuations':
            punctuations = "?!;.？！；。"
            res = []
            before = 0
            for i in range(len(sentence)):
                if sentence[i] in punctuations:
                    res.append(sentence[before:i + 1])
                    before = i + 1

        return res  # -> [分句1,...]


if __name__ == '__main__':
    # s = Separator(option='by_punctuations')
    s = Separator(option='by_model')
    r = s.separate("绝缘子、隔离开关及瓷套管应在安装前进行哪些试验？绝缘子、隔离开关及瓷套管应在安装前进行哪些试验。")
    print(r)
