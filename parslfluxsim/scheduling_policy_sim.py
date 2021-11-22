from parslfluxsim.workitem_sim import WorkItem

class Policy(object):
    def __init__ (self, name):
        self.name = name
        self.newworkitemqueue = []
        self.cpuqueue = []
        self.gpuqueue = []
        self.resubmitcpuqueue = []
        self.resubmitgpuqueue = []

    def add_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        pass

    def add_back_workitem (self, resourcetype, workitem):
        #print ('add_back_workitem ():')

        #print (resourcetype, workitem.get_id ())
        if resourcetype == 'CPU':
            self.gpuqueue.append (workitem)
            #print (self.gpuqueue)
        else:
            self.cpuqueue.append (workitem)
            #print (self.cpuqueue)

    def pop_resubmit_workitem (self, resourcetype):
        #print ('pop_resubmit_workitem():', resourcetype)

        if resourcetype == 'CPU':
            if len (self.resubmitcpuqueue) > 0:
                item = self.resubmitcpuqueue.pop (0)
                return item
            else:
                #print ('None')
                return None
        else:
            if len (self.resubmitgpuqueue) > 0:
                item = self.resubmitgpuqueue.pop (0)
                return item
            else:
                #print ('None')
                return None

    def pop_pending_workitem (self, resourcetype):
        #print ('pop_pending_workitem ():', resourcetype)
        if resourcetype == 'CPU':
            if len (self.gpuqueue) > 0:
                item = self.gpuqueue.pop (0)
                return item
            else:
                #print ('None')
                return None
        else:
            if len (self.cpuqueue) > 0:
                item = self.cpuqueue.pop (0)
                return item
            else:
                #print ('None')
                return None

    def get_pending_workitems (self, resourcetype):
        #print ('get_pending_workitems ():', resourcetype)

        if resourcetype == 'CPU':
            return self.gpuqueue.copy ()
        else:
            return self.cpuqueue.copy ()

    def get_new_workitem(self, resourcetype):
        new_workitem = None
        if len (self.newworkitemqueue) > 0:
            if resourcetype == self.newworkitemqueue[0].resourcetype:
                new_workitem = self.newworkitemqueue.pop(0)
        return new_workitem

    def get_pending_workitems_count (self, resourcetype):
        #print ('get_pending_workitems_count ():')
        if resourcetype == 'CPU':
            #print (resourcetype, len (self.gpuqueue))
            return len (self.gpuqueue)

        if resourcetype == 'GPU':
            #print (resourcetype, len (self.cpuqueue))
            return len (self.cpuqueue)

    def create_workitem_full (self, imanager, pmanager, resource_id):
        #print ('create_workitem_full ():')

        pipelinestages = pmanager.get_all_pipelinestages ()

        if imanager.get_remaining_count () == 0:
            return None

        images = imanager.get_images (1)

        image_key = list (images.keys())[0]

        new_workitem = WorkItem (image_key, images[image_key], None, \
                                 pipelinestages, resource_id, '', \
                                 0, '')

        return new_workitem

    def create_workitem (self, imanager, pmanager, resource_id, resourcetype):
        #print ('create_workitem ():', resourcetype)

        pipelinestages = pmanager.get_pipelinestages (None, resourcetype)
        if pipelinestages == None:
            #print ('None')
            return None

        if imanager.get_remaining_count () == 0:
            #print ('None')
            return None

        images = imanager.get_images (1)

        image_key = list (images.keys())[0]

        new_workitem = WorkItem (image_key, images[image_key], None, \
                                 pipelinestages, resource_id, resourcetype, \
                                 0, '')

        self.newworkitemqueue.append(new_workitem)

        return new_workitem

    #*******#
    def remove_complete_workitem (self, resource, pmanager, env):
        #print ('remove_complete_workitem ():', resource.id)
        cpu_workitem = resource.pop_if_complete ('CPU')

        if cpu_workitem != None:
            #print (cpu_workitem.print_data ())
            if cpu_workitem.get_status () == 'SUCCESS':
                #print ('adding to cpuqueue')
                self.cpuqueue.append (cpu_workitem)
                print (cpu_workitem.id)
                pmanager.remove_executor(cpu_workitem, resource)
                pmanager.add_workitem_queue (cpu_workitem, env.now)
            else:
                #print ('adding to resubmitcpuqueue')
                self.resubmitcpuqueue.append (cpu_workitem)

        gpu_workitem = resource.pop_if_complete ('GPU')

        if gpu_workitem != None:
            #print (gpu_workitem.print_data ())
            if gpu_workitem.get_status () == 'SUCCESS':
                #print ('adding to gpuqueue')
                self.gpuqueue.append (gpu_workitem)
                print(gpu_workitem.id)
                pmanager.remove_executor(gpu_workitem, resource)
                pmanager.add_workitem_queue(gpu_workitem, env.now)
            else:
                #print ('adding to resubmitgpuqueue')
                self.resubmitgpuqueue.append (gpu_workitem)


    def sort_complete_workitems_by_earliest_schedule_time (self, resourcetype):
        if resourcetype == 'CPU':
            self.resubmitcpuqueue = sorted (self.resubmitcpuqueue, key=lambda x:x.scheduletime)
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:x.scheduletime)
            #print (self.resubmitcpuqueue)
            #print (self.gpuqueue)
        else:
            self.resubmitgpuqueue = sorted (self.resubmitgpuqueue, key=lambda x:x.scheduletime)
            self.cpuqueue = sorted (self.cpuqueue, key=lambda x:x.scheduletime)
            #print (self.resubmitgpuqueue)
           # print (self.cpuqueue)

    def sort_complete_workitems_by_earliest_finish_time (self, resourcetype):
        if resourcetype == 'CPU':
            self.resubmitcpuqueue = sorted (self.resubmitcpuqueue, key=lambda x:x.endtime)
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:x.endtime)
            #print (self.resubmitcpuqueue)
            #print (self.gpuqueue)
        else:
            self.resubmitgpuqueue = sorted (self.resubmitcpuqueue, key=lambda x:x.endtime)
            self.cpuqueue = sorted (self.cpuqueue, key=lambda x:x.endtime)
            #print (self.resubmitgpuqueue)
            #print (self.cpuqueue)

    def sort_complete_workitems_by_latest_finish_time (self, resourcetype):
        if resourcetype == 'CPU':
            self.resubmitcpuqueue = sorted (self.resubmitcpuqueue, key=lambda x:x.endtime, reverse=True)
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:x.endtime, reverse=True)
            #print (self.resubmitcpuqueue)
            #print (self.gpuqueue)
        else:
            self.resubmitgpuqueue = sorted (self.resubmitcpuqueue, key=lambda x:x.endtime, reverse=True)
            self.cpuqueue = sorted (self.cpuqueue, key=lambda x:x.endtime, reverse=True)
            #print (self.resubmitgpuqueue)
            #print (self.cpuqueue)
