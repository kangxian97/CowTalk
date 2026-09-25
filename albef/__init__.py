"""Multimodal BERT used to fuse shape latents with text instructions.

``xbert.py`` is Hugging Face BERT (Apache-2.0) with the cross-attention changes
from Salesforce ALBEF (BSD-3-Clause, see LICENSE.txt). CowTalk uses ``BertModel``
in ``multi_modal`` mode; the rest of ALBEF is not included.
"""

from .xbert import BertConfig, BertEncoder, BertModel

__all__ = ["BertConfig", "BertEncoder", "BertModel"]
