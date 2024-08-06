#
# Copyright (c) 2023 HES-XPLAIN
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

from omnixai.explanations.base import ExplanationBase


class DimlpfidexExplanation(ExplanationBase):

    def __init__(self, mode) -> None:
        super().__init__()
        self.mode = mode
        self.explanations = {}
