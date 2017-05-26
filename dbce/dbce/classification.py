
"""
dbce.classification - classification logic

Copyright (c) 2016 Rafal Lesniak

This software is licensed as described in the file LICENSE.

"""

from dbce.constants import (DBCE_DIFF_CUR, DBCE_DIFF_UP, DBCE_DIFF_DOWN,
                            DBCE_DIFF_PREV, DBCE_DIFF_NEXT, DBCE_BITS_TO_CLASS)

def classify_block(versions, bits):
    if bits[0] == 0:
        raise ValueError("bits {}: invalid marker, current version should be always present".format(
            bits))
    return DBCE_BITS_TO_CLASS[tuple(versions)][bits]

def marker_to_bits(marker=None):
    """Return bit touple based on markings set

    The bits order is (CUR, UP, DOWN, PREV, NEXT).

    Example usage:
    >>> a = [DBCE_DIFF_CUR, DBCE_DIFF_UP]
    >>> marker_to_bits(a)
    (1, 1, 0, 0, 0)
    """

    # generate initial bit array filled with 0
    bits = [0] * 5
    if not marker:
        return tuple(bits)

    if DBCE_DIFF_CUR in marker:
        bits[0] = 1
    if DBCE_DIFF_UP in marker:
        bits[1] = 1
    if DBCE_DIFF_DOWN in marker:
        bits[2] = 1
    if DBCE_DIFF_PREV in marker:
        bits[3] = 1
    if DBCE_DIFF_NEXT in marker:
        bits[4] = 1

    return tuple(bits)
