from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self):
        pass

