import sys
import plugins
import os

def node_allocate(scheduler_pid, originentityvalue):
    print ('requesting new resource')
    pass

def node_restart():
    print ('restarting resource')
    pass

@plugins.register
def handle_exception(**kwargs):
    print (kwargs, os.getcwd ())
    op = kwargs['op']
    if op == 'allocate':
        node_allocate ()
    elif op == 'restart':
        node_restart ()
