from models.GTRSignalCore import SignalAugmentedModel

class Model(SignalAugmentedModel):
    def __init__(self, configs):
        super().__init__(configs, mode='high')
