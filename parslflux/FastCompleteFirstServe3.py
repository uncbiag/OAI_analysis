import datetime

from parslflux.resources import ResourceManager, Resource
from parslflux.input import InputManager2
from parslflux.pipeline import PipelineManager
from parslflux.workitem import WorkItem

from parslflux.scheduling_policy import Policy

class FastCompleteFirstServe3 (Policy):
    def add_new_workitems (self, rmanager, imanager, pmanager, empty_resources, resourcetype):
        #print ('add_new_workitems ():')

        rankings = {}

        workitems_needed = len (empty_resources)

        workitems = []
        workitems_dict = {}
        pending_workitems_dict = {}

        #print (workitems_needed)

        #print (empty_resources)

        pending_workitems = self.get_pending_workitems (resourcetype)

        #print (pending_workitems)


        total_done = 0

        for pending_workitem in pending_workitems:
            #print (pending_workitem)
            self.pop_pending_workitem (resourcetype)
            next_workitem = pending_workitem.compose_next_workitem (pmanager, None, resourcetype)
            if next_workitem != None:
                workitems.append (next_workitem)
                workitems_dict[next_workitem.get_id ()] = next_workitem
                total_done += 1
                pending_workitems_dict[next_workitem.get_id ()] = pending_workitem

        for workitem in workitems:
            workitem_pipelinestages = workitem.get_pipelinestages ()
            encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

            resources = rmanager.get_resources ()
            exectimes = {}
            for resource in resources:
                exectimes[resource.id] = resource.get_exectime (encoded_workitem_pipelinestages)

            sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

            rankings[workitem.get_id ()] = sorted_exec_times

        #print (rankings)

        if total_done < workitems_needed:
            additional_workitems = []

            while total_done < workitems_needed:
                new_workitem = self.create_workitem (imanager, pmanager, None, resourcetype)
                if new_workitem != None:
                    additional_workitems.append (new_workitem)
                    workitems_dict[new_workitem.get_id ()] = new_workitem
                    total_done += 1
                else:
                    break

            for workitem in additional_workitems:
                workitem_pipelinestages = workitem.get_pipelinestages ()
                encoded_workitem_pipelinestages = pmanager.encode_pipeline_stages (workitem_pipelinestages)

                resources = rmanager.get_resources ()
                exectimes = {}
                for resource in resources:
                    exectimes[resource.id] = resource.get_exectime (encoded_workitem_pipelinestages)

                sorted_exec_times = dict(sorted(exectimes.items(), key=lambda item: item[1]))

                rankings[workitem.get_id ()] = sorted_exec_times

        rankings = dict (sorted (rankings.items(), key = lambda item:list (item[1].values ())[0], reverse=True))

        if total_done < workitems_needed:
            workitems_needed = total_done

        #print (rankings)
        #print (workitems_needed)

        resources_done = {}
        workitems_done = {}

        resources = rmanager.get_resources ()
        for resource in resources:
            if resource in empty_resources:
                resources_done[resource.id] = False
            else:
                resources_done[resource.id] = True

        for workitem_id in workitems_dict:
            workitems_done[workitem_id] = False

        total_workitems_done = 0

        while True:
            best_rankings = {}

            for workitem_id in rankings:
                #print (workitem_id)

                if workitems_done[workitem_id] == True:
                    #print (workitem_id, 'done')
                    continue

                resource_rankings = rankings[workitem_id]

                #print (resource_rankings)

                index = 0
                for resource_id in resource_rankings:
                    if resources_done[resource_id] == True:
                        #print (resource_id, 'done')
                        continue

                    resource = rmanager.get_resource (resource_id)

                    if resource in empty_resources:
                        best_rankings[workitem_id] = [index, resource_id]
                        break
                    index += 1

            #print (best_rankings)

            if len (best_rankings) == 0:
                #print ('all done')
                break

            best_ranking = None
            best_ranking_workitem_id = None

            for workitem_id in best_rankings:
                ranking = best_rankings[workitem_id][0]

                if best_ranking == None:
                    best_ranking = best_rankings[workitem_id]
                    best_ranking_workitem_id = workitem_id
                elif best_ranking[0] >= ranking:
                    best_ranking = best_rankings[workitem_id]
                    best_ranking_workitem_id = workitem_id

            workitem = workitems_dict[best_ranking_workitem_id]
            workitem.set_resource_id (best_ranking[1])
            resource = rmanager.get_resource (best_ranking[1])
            resource.add_workitem (workitem, resourcetype)

            workitems_done[best_ranking_workitem_id] = True
            resources_done[best_ranking[1]] = True

            total_workitems_done += 1

            if total_workitems_done == workitems_needed:
                break


        for workitem_id in rankings:
            if workitems_done[workitem_id] == True:
                continue
            workitem = workitems_dict[workitem_id]
            if workitem.get_id () in pending_workitems_dict:
                pending_workitem = pending_workitems_dict[workitem.get_id ()]
                self.add_back_workitem (resourcetype, pending_workitem)
