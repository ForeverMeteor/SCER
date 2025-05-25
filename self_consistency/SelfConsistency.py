from abc import ABC, abstractmethod


class SelfConsistency(ABC):
    def __init__(self, self_consistency_rounds):
        self.self_consistency_rounds = self_consistency_rounds

    @abstractmethod
    def get_inference_paths(self, question, test=False):
        pass
