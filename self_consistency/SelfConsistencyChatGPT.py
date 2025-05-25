from openai import OpenAI

import CONSTANT
from CONSTANT import *
from SelfConsistency import SelfConsistency


class SelfConsistencyChatGPT(SelfConsistency):
    def __init__(self, self_consistency_rounds):
        super().__init__(self_consistency_rounds)

        self.client = OpenAI(
            base_url=CONSTANT.GET_URL(),
            api_key=CONSTANT.GET_KEY(),
        )
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
                paths.append("Yes. The official language of Saint Vincent and the Grenadines is English. So the answer is yes.")
                print("Round {} finished".format(i + 1))

        else:
            for i in range(self.self_consistency_rounds):
                # print("Round {}".format(i+1))
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": self.CoT + question + "\nA:"
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
                res = chat_completion.choices[0].message.content
                paths.append(res)
                # print("Round {} finished".format(i + 1))

        print("LLM Finished responding.")
        return paths  # -> [str,...]


if __name__ == '__main__':
    self_consistency = SelfConsistencyChatGPT(5)
    p = self_consistency.get_inference_paths(question="Is the language used in Saint Vincent and the Grenadines rooted in English?",
                                             test=True)
    print(p)
