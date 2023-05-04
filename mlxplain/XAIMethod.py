"""
The base XAI method class
"""

import numpy as np
import pandas as pd

class XAIMethod:
    def __init__(self, data):
        self.data = data

    def explain(self):
        raise NotImplementedError("Method not implemented")

