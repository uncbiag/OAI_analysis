#!/usr/bin/env python3.6
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


def read_pid_from_file(pid_file):
    """
    get pid from a text file in format
    ```
    index PID
    1 XXXX
    2 XXXX
    3 XXXX
    ```
    :param pid_file:
    :return:
    """
    pids = []
    with open(pid_file) as f:
        lines = f.readlines()
        for l in lines[1:]:
            pids.append(int(l.strip().split(' ')[1]))

    return pids
