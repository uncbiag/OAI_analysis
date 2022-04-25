from data.OAI_data import OAIData, OAIImage, OAIPatients

import yaml
import pickle
import sys

class Input:
    def __init__ (self, id, name, location):
        self.id = id
        self.location = location
        self.name = name

    def get_name (self):
        return self.name

    def get_location (self):
        return self.location

    def get_id (self):
        return self.id

    def print_data (self):
        print (self.id, self.name, self.location)

class InputManager:
    def __init__ (self, inputfile):
        self.inputfile = inputfile
        self.inputdata = []
        self.index = 0

    def parse_input (self):
        inputdata = pickle.load (open (self.inputfile, "rb" ))
        for input in inputdata:
            self.inputdata.append (Input(input[0], input[1], input[2]))

    def get_input (self, count):
        data = self.inputdata[self.index:self.index + count]
        self.index = self.index + count
        return data

    def print_data (self):
        for i in self.inputdata:
            i.print_data ()

class InputManager2:
    def __init__(self, config_file):
        self.index = 0
        self.completion_status = {}

        config_data_file = open (config_file)

        config_data = yaml.load (config_data_file, Loader=yaml.FullLoader)

        OAI_data_sheet = config_data['oai_data_sheet']
        OAI_data_directory = config_data['oai_data_directory']

        OAI_patients_path = config_data['oai_enrollees']

        print (OAI_data_sheet, OAI_data_directory, OAI_patients_path)

        self.OAI_data = OAIData (OAI_data_sheet, OAI_data_directory)
        self.oai_patients = OAIPatients (OAI_patients_path)

        self.analysis_patients = list (self.OAI_data.patient_set)

        print (self.analysis_patients)

        self.knee_type = ['LEFT_KNEE', 'RIGHT_KNEE']
        self.time_point = 12

        self.analysis_images = self.OAI_data.get_images(patient_id = self.analysis_patients,
                                                        part = self.knee_type,
                                                        visit_month = [self.time_point])

        print (len (self.analysis_images))


    def get_total_count (self):
        return len (self.analysis_images)

    def get_images (self, count):
        return_images = {}

        if count > len(self.analysis_images) - self.index:
            count = len(self.analysis_images) - self.index

        images = self.analysis_images[self.index:self.index + count]

        for image in images:
            return_images[str(image.name)] = image.get_raw_info()
        self.index = self.index + count

        #print(return_images)

        return return_images

    def add_back_images (self, count):
        self.index -= count

    def get_remaining_count (self):
        return len(self.analysis_images) - self.index

    def print_data (self):
        for image in self.analysis_images:
            print (image)

    def set_complete(self, image_id, version, status):
        if image_id not in self.completion_status:
            self.completion_status[image_id] = {}

        self.completion_status[image_id][str(version)]= status

        #print('set_complete', image_id, self.completion_status[image_id])

    def is_complete(self, image_id, version):
        if str(version) in self.completion_status[image_id]:
            return self.completion_status[image_id][str(version)]
        else:
            return False

if __name__ == "__main__":
    configfile = sys.argv[1]
    i = InputManager2 (configfile)
    i.print_data()
