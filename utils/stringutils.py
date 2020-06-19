import re


def slugify(string):
    """
    slugifies a string
    :param string:
    :return:
    """
    string = string.lower()
    string = re.sub(r"\s+", "_", string)
    string = re.sub(r"'", "_", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"/", "_", string)
    return string


def binary_seq_to_decimal(iterable):
    seqlen = len(iterable)
    val = 0
    for idx, itm in enumerate(iterable):
        itm = int(itm)
        if itm == 0:
            continue
        val += 2 ** (seqlen - idx - 1)
    return val
