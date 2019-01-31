#!/usr/bin/env python
"""
Operations on strings
Created by zhenlinx on 1/19/19
"""



def replace_extension(name: str, string: str):
    """
    Replace the file extension with a string
    E.g. /a/b/file.txt -> /a/b/file_ext.zip
    :param name:
    :return: The new path of the file with extended name
    """
    preflix = '.'.join(name.split('.')[:-1])
    return "{}{}".format(preflix, string)
