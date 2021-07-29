import sys
import plugins

def workitem_cancel():
    pass

def workitem_reschedule():
    pass

@plugins.register
def handle_exception(entity, op):
    print ('workitem resource')
