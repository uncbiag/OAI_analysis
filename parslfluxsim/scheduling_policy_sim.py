from parslfluxsim.workitem_sim import WorkItem
import copy

class Policy(object):
    def __init__ (self, name, pmanager):
        self.name = name
        self.newworkitemqueue = []
        self.cpuqueue = []
        self.gpuqueue = []
        self.resubmitcpuqueue = []
        self.resubmitgpuqueue = []
        self.pmanager = pmanager

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

    def pop_pending_workitem_by_id (self, resourcetype, workitem_id):
        #print ('pop_pending_workitem_by_id', workitem_id)
        if resourcetype == 'CPU':
            if len (self.gpuqueue) > 0:
                #for workitem in self.gpuqueue:
                #    print ('pop_pending_workitem_id before', 'GPU', workitem, workitem.id)
                index = 0
                for workitem in self.gpuqueue:
                    if workitem.id == workitem_id:
                        break
                    index += 1
                #print ('pop_pending_workitem_id', 'GPU', index, len(self.gpuqueue), self.gpuqueue[index])
                if index < len (self.gpuqueue):
                    item = self.gpuqueue.pop(index)
                    #for workitem in self.gpuqueue:
                    #    print('pop_pending_workitem_id after', 'GPU', workitem, workitem.id)
                    return item
                else:
                    return None
            else:
                return None
        else:
            if len (self.cpuqueue) > 0:
                #for workitem in self.cpuqueue:
                #    print ('pop_pending_workitem_id before', 'CPU', workitem, workitem.id)
                index = 0
                for workitem in self.cpuqueue:
                    if workitem.id == workitem_id:
                        break
                    index += 1
                #print('pop_pending_workitem_id', 'CPU', index, len(self.cpuqueue), self.cpuqueue[index])
                if index < len (self.cpuqueue):
                    item = self.cpuqueue.pop(index)
                    #for workitem in self.cpuqueue:
                    #    print('pop_pending_workitem_id before', 'CPU', workitem, workitem.id)
                    return item
                else:
                    return None
            else:
                return None

    def get_pending_workitems (self, resourcetype):
        #print ('get_pending_workitems ():', resourcetype)

        if resourcetype == 'CPU':
            newcopy = copy.deepcopy (self.gpuqueue)
            return newcopy
        else:
            newcopy = copy.deepcopy (self.cpuqueue)
            return newcopy

    def get_pending_workitems_no_copy (self, resourcetype):
        if resourcetype == 'CPU':
            return self.gpuqueue
        else:
            return self.cpuqueue

    def get_new_workitem(self, resourcetype):
        new_workitem = None
        if len (self.newworkitemqueue) > 0:
            if resourcetype == self.newworkitemqueue[0].resourcetype:
                new_workitem = self.newworkitemqueue.pop(0)
        return new_workitem

    def add_back_new_workitem (self, workitem):
        self.newworkitemqueue.insert(0, workitem)

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

        pipelinestage = pmanager.get_pipelinestage (None, resourcetype)
        if pipelinestage == None:
            #print ('None')
            return None

        if imanager.get_remaining_count () == 0:
            #print ('None')
            return None

        images = imanager.get_images (1)

        image_key = list (images.keys())[0]

        new_workitem = WorkItem (image_key, images[image_key], None, \
                                 pipelinestage, resource_id, resourcetype, \
                                 0, '')

        self.newworkitemqueue.append(new_workitem)

        return new_workitem


    def remove_complete_workitem (self, resource, pmanager, env):
        #print ('remove_complete_workitem ():', resource.id)
        cpu_workitem = resource.pop_if_complete ('CPU')

        if cpu_workitem != None:
            #print (cpu_workitem.print_data ())
            if cpu_workitem.get_status () == 'SUCCESS':
                #print ('adding to cpuqueue')
                if int(cpu_workitem.version) < len (self.pmanager.pipelinestages) - 1:
                    self.cpuqueue.append (cpu_workitem)
                pmanager.remove_executor(cpu_workitem, resource, env.now)
                pmanager.add_workitem_queue (cpu_workitem, env.now)
            else:
                #print ('adding to resubmitcpuqueue')
                self.resubmitcpuqueue.append (cpu_workitem)

        gpu_workitem = resource.pop_if_complete ('GPU')

        if gpu_workitem != None:
            #print (gpu_workitem.print_data ())
            if gpu_workitem.get_status () == 'SUCCESS':
                #print ('adding to gpuqueue')
                if int(gpu_workitem.version) < len(self.pmanager.pipelinestages) - 1:
                    self.gpuqueue.append (gpu_workitem)
                pmanager.remove_executor(gpu_workitem, resource, env.now)
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

    def sort_complete_workitems_by_phase_and_stage (self, resourcetype):
        #print ('sort_complete_workitems_by_stage_id ()')
        if resourcetype == 'CPU':
            #print('CPU')
            self.resubmitcpuqueue = sorted (self.resubmitcpuqueue, key=lambda x:(x.phase_index, x.version))
            if len (self.gpuqueue) >= 2:
                for item in self.gpuqueue:
                    item.print_data ()
            #print ('$$$$$$$$$$$$$$')
            self.gpuqueue = sorted (self.gpuqueue, key=lambda x:(x.phase_index, x.version))
            if len (self.gpuqueue) >= 2:
                for item in self.gpuqueue:
                    item.print_data ()
            #print ('##############')
        else:
            #print('GPU')
            self.resubmitgpuqueue = sorted(self.resubmitgpuqueue, key=lambda x: (x.phase_index, x.version))
            if len (self.cpuqueue) >= 2:
                for item in self.cpuqueue:
                    item.print_data()
            #print('$$$$$$$$$$$$$$$')
            self.cpuqueue = sorted(self.cpuqueue, key=lambda x: (x.phase_index, x.version))
            if len (self.cpuqueue) >= 2:
                for item in self.cpuqueue:
                    item.print_data()
            #print('##############')

    def sort_complete_workitems_by_stage (self, resourcetype):
        # print ('sort_complete_workitems_by_stage_id ()')
        if resourcetype == 'CPU':
            # print('CPU')
            self.resubmitcpuqueue = sorted(self.resubmitcpuqueue, key=lambda x: (x.version), reverse=True)
            if len(self.gpuqueue) >= 2:
                for item in self.gpuqueue:
                    item.print_data()
            # print ('$$$$$$$$$$$$$$')
            self.gpuqueue = sorted(self.gpuqueue, key=lambda x: (x.version), reverse=True)
            if len(self.gpuqueue) >= 2:
                for item in self.gpuqueue:
                    item.print_data()
            # print ('##############')
        else:
            # print('GPU')
            self.resubmitgpuqueue = sorted(self.resubmitgpuqueue, key=lambda x: (x.version), reverse=True)
            if len(self.cpuqueue) >= 2:
                for item in self.cpuqueue:
                    item.print_data()
            # print('$$$$$$$$$$$$$$$')
            self.cpuqueue = sorted(self.cpuqueue, key=lambda x: (x.version), reverse=True)
            if len(self.cpuqueue) >= 2:
                for item in self.cpuqueue:
                    item.print_data()
            # print('##############')

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
