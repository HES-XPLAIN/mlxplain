"""
The base tabular method class
"""

from hes_xplain.methods.XAIMethod import XAIMethod

class TabularXAIMethod(XAIMethod):
    def __init__(self, data):
        super().__init__(data)
        self.tabular_data = data

    def preprocess_data(self):
        raise NotImplementedError("Method not implemented")

