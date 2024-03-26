import pytest
from torch import rand
from network import Unet

class TestModel:
    def test_loading(self):
        pass
    def test_forward(self):
        image = rand(256, 256, 192)
        output = Unet.forward(image=image)
        assert self.image.shape == output.shape