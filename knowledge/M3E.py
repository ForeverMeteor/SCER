import os.path
from sentence_transformers import SentenceTransformer
import torch

import CONSTANT


class M3E:
    def __init__(self):
        self.model_path = os.path.join(CONSTANT.GET_PROJECT_ROOT(), "model", "moka-ai/m3e-base")
        self.model = SentenceTransformer(self.model_path)
        torch.cuda.set_device(int(CONSTANT.GET_CUDA().split(":")[-1]))

    def encode(self, sentences):
        # sentences = [sentence]
        return self.model.encode(sentences)


if __name__ == '__main__':
    sentences = [
        '无运动部件的电能变换器，它改变与电能相关联的电压及电流而不改变频率。',
        '电力行业各单位充分整合和利用现有资源，在建立和完善本单位“一案三制”的基础上，全面加强相关重要应急环节的建设，包括：监测预警、应急指挥、应急队伍、物资保障、培训演练、科技支撑、恢复重建等。注：“一案三制”指应急预案和应急体制、机制、法制。',
        ]

    m3e = M3E()
    embeddings = m3e.encode(sentences)
    print(embeddings)
