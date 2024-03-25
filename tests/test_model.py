import pytest
from torch import rand
from ..code.network import Unet

class TestModel(Unet):
    def __init__(self, image, learning_rate=0.001):
        super().__init__(learning_rate)
        self.image = rand(256, 256, 192)
    def test_loading(self):
        pass
    def test_forward(self):
        output = self.forward(self.image)
        assert self.image.shape == output.shape