import sys
import plugins
from worker import *

def node_allocate():
    print ('requesting new resource')
    pass

def node_restart():
    print ('restarting resource')
    pass

@plugins.register
def handle_exception(entity, op):
    if op == 'allocate':
        node_allocate ()
    elif op == 'restart':
        node_restart ()
