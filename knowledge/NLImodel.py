import os.path
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from CONSTANT import *


class NLI:
    def __init__(self):  # NLI模型
        self.small_batch_size = 1  # 2
        # 此处更换NLI模型的实现
        self.hg_model_hub_name = os.path.join("ynie", "albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")
        self.model_path = os.path.join(GET_PROJECT_ROOT(), "model", self.hg_model_hub_name)
        # print(self.model_path)
        self.entailment_tokenizer = AutoTokenizer.from_pretrained(
                                        self.model_path,
                                        cache_dir=os.path.join(GET_PROJECT_ROOT(), ".cache/transformers"))
        self.entailment_model = AutoModelForSequenceClassification.from_pretrained(
                                        self.model_path,
                                        cache_dir=os.path.join(GET_PROJECT_ROOT(), ".cache/transformers"))

    '''
        运行文本蕴含模型，改编自RR官方代码commonsense_evidence.py
        batch_premise_hypothesis -> [(随机游走串, 分句),...]
    '''
    def run_textual_entailment(self, batch_premise_hypothesis):
        # Note:
        # "id2label": {
        #     "0": "entailment",  # 蕴含
        #     "1": "neutral",  # 中性
        #     "2": "contradiction"  # 矛盾
        # },
        max_length = 512
        tokenized_input_seq_pair = self.entailment_tokenizer.batch_encode_plus(batch_premise_hypothesis,
                                                                               max_length=max_length,
                                                                               return_token_type_ids=True,
                                                                               truncation=True,
                                                                               return_tensors='pt',
                                                                               padding=True)
        input_ids = tokenized_input_seq_pair['input_ids']
        token_type_ids = tokenized_input_seq_pair['token_type_ids']
        attention_mask = tokenized_input_seq_pair['attention_mask']

        self.entailment_model.to(GET_CUDA())
        outputs = self.entailment_model(input_ids.to(GET_CUDA()),
                                        attention_mask=attention_mask.to(GET_CUDA()),
                                        token_type_ids=token_type_ids.to(GET_CUDA()),
                                        labels=None)
        predicted_probability = torch.softmax(outputs[0], dim=1).cpu().tolist()  # batch_size only one
        return predicted_probability


    '''
        获取蕴含分数，改编自RR官方代码commonsense_evidence.py
    '''
    def get_entailment_scores(self, sentence: str, random_walks: list):
        # sentence -> 分句，即ei;  random_walks->[随机路径1,...]
        predicted_probability_list = []  # -> [softmax(,,),...]
        # the small batch size should be adjusted to GPU memory size
        self.small_batch_size = 1  # 2
        batch_premise_hypothesis = [(random_walks[i], sentence) for i in range(len(random_walks))]

        batch_num = int(math.ceil(len(random_walks) / self.small_batch_size))
        for batch_idx in range(batch_num):
            # 模型于此处运行
            # batch_premise_hypothesis -> [(WikiPara, 分句),...]
            res = self.run_textual_entailment(
                batch_premise_hypothesis[batch_idx * self.small_batch_size: batch_idx * self.small_batch_size + self.small_batch_size]
            )  # -> (softmax)(蕴含分数,中性分数,矛盾分数)
            predicted_probability_list.extend(res)

        return predicted_probability_list  # -> [(softmax)(蕴含分数,中性分数,矛盾分数),...]
    # end def


if __name__ == '__main__':
    n = NLI()