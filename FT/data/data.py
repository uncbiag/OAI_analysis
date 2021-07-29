import sys
import plugins

def data_recover():
    pass

@plugins.register
def handle_exception(entity, op):
    print ('data resource')
