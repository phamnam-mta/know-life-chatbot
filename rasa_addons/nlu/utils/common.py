import os

from typing import Any, Dict

def is_duplicated(e1: Dict, e2: Dict):
    """
    check if 2 entities are shared the same index
    """
    return e1['start'] == e2['start'] and e1['end'] == e2['end']


def is_overlap(e1: Dict, e2: Dict):
    """
    check if 2 entities are overlapping in text index
    """
    return e1['start'] <= e2['start'] <= e1['end'] or e2['start'] <= e1['start'] <= e2['end']