import sys
import plugins

def workitem_cancel():
    pass

def workitem_reschedule():
    pass

@plugins.register
def handle_exception(**kwargs):
    print (kwargs)
    print ('workitem resource')
