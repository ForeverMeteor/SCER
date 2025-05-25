import re

from openai import OpenAI

import CONSTANT


class NER:
    def __init__(self):
        self.client = OpenAI(
                base_url=CONSTANT.GET_URL(),
                api_key=CONSTANT.GET_KEY(),
            )

    '''
        实体识别结果
        :return str {"entity":[...]}
        :author 
    '''
    def gpt_ner(self, text):
        # print("NER started")
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": """
                    你是一位电力领域的专家，请你按照要求完成电力实体识别的任务。\n
                    要求：\n
                    1. 抽取电力专业词汇；\n
                    2. 存在多个实体必须全部抽取出来；\n
                    3. 不存在实体直接返回“{"entity":[]}”；\n
                    4. 请返回JSON格式的结果，格式为“{"entity":[e1, e2, ...]}”\n
                    文本：\n""" + text,
                }
            ],
            model="gpt-3.5-turbo",
        )
        res = chat_completion.choices[0]
        # print(res)
        # print(type(res))
        """
            Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='{"entity":["电力系统","故障","异常情况","保护装置","保护系统","整个电力系统","个别构成单元","变压器"]}', role='assistant', function_call=None, tool_calls=None))
            <class 'openai.types.chat.chat_completion.Choice'>
            {"entity":["电力系统","故障","异常情况","保护装置","保护系统","整个电力系统","个别构成单元","变压器"]}
        """
        # print("NER finished")
        return res.message.content
        # return ["电力系统","故障","异常情况","保护装置","保护系统","整个电力系统","个别构成单元","变压器"]

    '''
        加这个函数的目的是防止返回结果有时候格式不太一样会抛错
        目前的实现还是简单的eval
    '''
    def trans_into_list(self, content):
        try:
            content = re.sub("\n+", "", content)
            content = re.sub("\r+", "", content)
            content = content.strip("\"")
            d = eval(content)
            # print(type(d))
            l = d["entity"]
        except Exception as e:
            print(e)
            print(content)
            return None
        return l


if __name__ == '__main__':
    ner = NER()
    # res = ner.gpt_ner("在电力系统中检出故障或其他异常情况，从而使故障切除、终止异常情况或发出信号或指示。注1：保护是一个用于保护装置或保护系统的一般性词语。注2：保护可以用于描述整个电力系统的保护或者电力系统中个别构成单元的保护，例如变压器保...")
    res = ner.trans_into_list('{"entity":["电力系统","故障","异常情况","保护装置","保护系统","整个电力系统","个别构成单元","变压器"]}')
    print(res)
    # print(type(res))

