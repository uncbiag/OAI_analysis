#!/usr/bin/env python3.6
"""
Created by zhenlinx on 1/31/19
"""
import os
from data.OAI_data import OAIData, OAIImage, OAIPatients
from oai_image_analysis import OAIImageAnalysis
from registration.registers import NiftyReg, AVSMReg
from segmentation.segmenter import Segmenter3DInPatchClassWise
import random
import shutil
from mpi4py import MPI

atlas_path=os.path.abspath("pipelines.py")
atlas_path=atlas_path.replace("pipelines.py","")
ATLAS_IMAGE_PATH=atlas_path+"atlas/atlas_60_LEFT_baseline_NMI/atlas.nii.gz"
print("ATLAS_IMAGE_PATH:"+ATLAS_IMAGE_PATH)
# ATLAS_FC_MESH_PATH = "/playpen/zhenlinx/Code/OAI_analysis/atlas/atlas_60_LEFT_baseline_NMI/atlas_FC_inner_mesh_world.ply"
# ATLAS_TC_MESH_PATH = "/playpen/zhenlinx/Code/OAI_analysis/atlas/atlas_60_LEFT_baseline_NMI/atlas_TC_inner_mesh_world.ply"
# ATLAS_FC_2D_MAP_PATH = "./data/FC_inner_optional_embedded.npy"
# ATLAS_TC_2D_MAP_PATH = "./data/TC_inner_optional_embedded.npy"
ATLAS_FC_MESH_PATH = os.path.join(os.getcwd(),"data/atlas_FC_inner_mesh_world.ply")
ATLAS_TC_MESH_PATH = os.path.join(os.getcwd(),"data/atlas_TC_inner_mesh_world.ply")
ATLAS_FC_2D_MAP_PATH = os.path.join(os.getcwd(), "data/FC_inner_embedded.npy")
ATLAS_TC_2D_MAP_PATH = os.path.join(os.getcwd(), "data/TC_inner_embedded.npy")

first_folder_names_list=["1.C.2","3.C.2","5.C.1","6.C.1","8.C.1"]
second_folder_names_list=["1.E.1","3.E.1""5.E.1","6.E.1","8.E.1"]
months_list=["12","24","36","48","72"]
file_name_list=["12m.zip","24m.zip","36m.zip","48m.zip","72m.zip"]
file_name_small_list=["12m","24m","36m","48m","72m"]

f=open("./config.txt","r")
for line in f:
    if "OAI_data_sheet" in line:
        data_sheet=line.split('=')
        OAI_data_sheet=data_sheet[1].strip()
        print("OAI_data_sheet:"+OAI_data_sheet)
    if "OAI_data" in line:
        data=line.split('=')
        OAI_data_dir=data[1].strip()
        print("OAI_data:"+OAI_data_dir)
    if "OAI_results" in line:
        results=line.split('=')
        OAI_results=results[1].strip()
        print("OAI_results:"+OAI_results)
    if "Month" in line:
        month=line.split('=')
        OAI_month=month[1].strip()
        month_index=months_list.index(OAI_month)
        OAI_month=int(OAI_month)
        print("Month:"+str(OAI_month))
f.close()

def build_default_analyzer(ckpoint_folder=None, use_nifty=True,avsm_path=None):
    niftyreg_path = "/home/uray/niftyreg/niftyreg-install/bin"
    avsm_path = avsm_path + '/demo'
    register = NiftyReg(niftyreg_path) if use_nifty else AVSMReg(avsm_path)
    if not ckpoint_folder:
        ckpoint_folder = "./segmentation/ckpoints/UNet_bias_Nifti_rescaled_LEFT_train1_patch_128_128_32_batch_4_sample_0.01-0.02_BCEWithLogitsLoss_lr_0.001/01272019_212723"
    segmenter_config = dict(
        ckpoint_path=os.path.join(ckpoint_folder, "model_best.pth.tar"),
        training_config_file=os.path.join(ckpoint_folder, "train_config.json"),
        device="cuda",
        batch_size=2,
        overlap_size=(16, 16, 8),
        output_prob=True,
        output_itk=True,
    )
    affine_config = dict(smooth_moving=-1, smooth_ref=-1,
                         max_iterations=10,
                         pv=30, pi=30,
                         num_threads=30)
    bspline_config = dict(
        max_iterations=300,
        # num_levels=3, performed_levels=3,
        smooth_moving=-1, smooth_ref=0,
        sx=4, sy=4, sz=4,
        num_threads=32,
        be=0.1,  # bending energy, second order derivative of deformations (0.01)
    )

    segmenter = Segmenter3DInPatchClassWise(mode="pred", config=segmenter_config)
    analyzer = OAIImageAnalysis(use_nifty)
    analyzer.set_atlas(atlas_image_file=ATLAS_IMAGE_PATH, atlas_FC_mesh_file=ATLAS_FC_MESH_PATH,
                       atlas_TC_mesh_file=ATLAS_TC_MESH_PATH)
    analyzer.set_register(register=register, affine_config=affine_config, bspline_config=bspline_config)
    analyzer.set_segmenter(segmenter=segmenter)
    analyzer.set_preprocess(bias_correct=False, reset_coord=True, normalize_intensity=True, flip_to="LEFT")
    return analyzer


def demo_analyze_single_image(use_nifti,avsm_path=None,do_clean=False):
    OAI_data = OAIData(OAI_data_sheet, OAI_data_dir)
    OAI_data.set_processed_data_paths(OAI_results,None if use_nifti else 'avsm')
    test_image = OAI_data.get_images(patient_id= [9298954])[0] # 9279291, 9298954,9003380
    analyzer = build_default_analyzer(use_nifty=use_nifti, avsm_path=avsm_path)
    analyzer.preprocess(test_image, overwrite=False)
    analyzer.segment_image_and_save_results(test_image, overwrite=False)
    analyzer.close_segmenter()
    analyzer.extract_surface_mesh(test_image, overwrite=False)
    analyzer.register_image_to_atlas(test_image, True)
    analyzer.warp_mesh(test_image, overwrite=True,do_clean=do_clean)
    analyzer.project_thickness_to_atlas(test_image, overwrite=False)
    analyzer.set_atlas_2D_map(ATLAS_FC_2D_MAP_PATH,ATLAS_TC_2D_MAP_PATH)
    analyzer.compute_atlas_2D_map(n_jobs=None)
    analyzer.project_thickness_to_2D(test_image, overwrite=False)
    analyzer.eval_registration_surface_distance(test_image)
    analyzer.get_surface_distances_eval()


def demo_analyze_cohort(use_nifti,avsm_path=None, do_clean=False):
    comm=MPI.COMM_WORLD
    OAI_data_sheet_MPI=OAI_data_sheet+"."+MPI.Get_processor_name()
    OAI_data = OAIData(OAI_data_sheet_MPI, OAI_data_dir)
    OAI_data.set_processed_data_paths(OAI_results,None if use_nifti else 'avsm')

    patients_ASCII_file_path = "data/Enrollees.txt"
    oai_patients = OAIPatients(patients_ASCII_file_path)
    progression_cohort_patient = oai_patients.filter_patient(V00COHORT='1: Progression')

    progression_cohort_patient_6visits = list(progression_cohort_patient & OAI_data.patient_set)
    progression_cohort_images = OAI_data.get_images(visit_month=[OAI_month])

    subcohort_images = progression_cohort_images
    analyzer = build_default_analyzer(use_nifty=use_nifti, avsm_path=avsm_path)

    analyzer.preprocess_parallel(image_list=subcohort_images, n_workers=32, overwrite=True)
    for test_image in subcohort_images:
        analyzer.segment_image_and_save_results(test_image, overwrite=True)
    analyzer.close_segmenter()

    for i, test_image in enumerate(subcohort_images):
        print("\n[{}] {}\n".format(i, test_image.name))
        analyzer.register_image_to_atlas(test_image, True)
        analyzer.extract_surface_mesh(test_image, overwrite=True)
        analyzer.warp_mesh(test_image, overwrite=True,do_clean=do_clean)
        analyzer.eval_registration_surface_distance(test_image)
        analyzer.set_atlas_2D_map(ATLAS_FC_2D_MAP_PATH, ATLAS_TC_2D_MAP_PATH)
        analyzer.compute_atlas_2D_map(n_jobs=None)
        analyzer.project_thickness_to_atlas(test_image, overwrite=True)
        analyzer.project_thickness_to_2D(test_image, overwrite=True)

    analyzer.get_surface_distances_eval()

def parse_csv():
    first_folder=first_folder_names_list[month_index]
    second_folder=second_folder_names_list[month_index]

    first_set=set()
    second_set=set()

    first_list=[]
    second_list=[]

    f_parsed_csv=open("./parsed_csv.csv","w")
    with open("./data/SEG_3D_DESS_6visits.csv") as f_original_csv:
        first_line = f_original_csv.readline()
        f_parsed_csv.write(first_line)
    f_parsed_csv.close()

    f_original_csv=open("./data/SEG_3D_DESS_6visits.csv","r")
    f_parsed_csv=open("./parsed_csv.csv","a")
    for line in f_original_csv:
        if first_folder in line:
            f_parsed_csv.write(line)
            split_1=line.split(first_folder+"/")
            split_1_1=split_1[1].split("/")
            first_set.add(split_1_1[0])
        if second_folder in line:
            f_parsed_csv.write(line)
            split_2=line.split(second_folder+"/")
            split_2_1=split_2[1].split("/")
            second_set.add(split_2_1[0])
    f_original_csv.close()
    f_parsed_csv.close()

    first_list=list(first_set)
    second_list=list(second_set)

    f_first_list=open("./First_Folder_List.txt","w")
    for i in range(len(first_list)):
        f_first_list.write(first_list[i]+"\n")
    f_first_list.close()

    f_second_list=open("./Second_Folder_List.txt","w")
    for i in range(len(second_list)):
        f_second_list.write(second_list[i]+"\n")
    f_second_list.close()

def create_csv():
    file_name=file_name_list[month_index]
    file_name_small=file_name_small_list[month_index]

    folder_1=first_folder_names_list[month_index]
    folder_2=second_folder_names_list[month_index]

    list_folder_1=[]
    list_folder_2=[]
    list_folder_1_flag=[]
    list_folder_2_flag=[]
    set1=()
    set2=()
    f_first=open("./First_Folder_List.txt","r")
    for line in f_first:
        line=line.rstrip('\n')
        list_folder_1.append(line)
        list_folder_1_flag.append("no")
    f_first.close()

    f_second=open("./Second_Folder_List.txt","r")
    for line in f_second:
        line=line.rstrip('\n')
        list_folder_2.append(line)
        list_folder_2_flag.append("no")
    f_second.close()


    comm=MPI.COMM_WORLD
    folder_1_partition = int((len(list_folder_1))/comm.size)
    folder_2_partition = int((len(list_folder_2))/comm.size)

    folder_1_remaining = ((len(list_folder_1))%comm.size)
    folder_2_remaining = ((len(list_folder_2))%comm.size)

    stride_1 = comm.rank*folder_1_partition
    stride_2 = comm.rank*folder_2_partition
    f_tmp_csv=open(OAI_data_sheet+"."+MPI.Get_processor_name(),"w")
    with open('parsed_csv.csv') as f_sample_csv:
        first_line = f_sample_csv.readline()
        f_tmp_csv.write(first_line)
    f_tmp_csv.close()
    set_k=set()
    for k in range(stride_1,((comm.rank*folder_1_partition)+folder_1_partition)):
        f_parsed_csv=open("./parsed_csv.csv","r")
        f_tmp_csv=open(OAI_data_sheet+"."+MPI.Get_processor_name(),"a")
        for line in f_parsed_csv:
            if list_folder_1[k] in line and folder_1 in line:
                f_tmp_csv.write(line)
                set_k.add(list_folder_1[k])
                if(list_folder_1_flag[k] is "no"):
                    list_folder_1_flag[k]="yes"
        f_parsed_csv.close()
        f_tmp_csv.close()

    set_m=set()
    if folder_1_remaining > 0:
        for m in range((((comm.size-1)*folder_1_partition)+folder_1_partition),len(list_folder_1)):
            if ((len(list_folder_1))%m)==comm.rank:
                f_parsed_csv=open("./parsed_csv.csv","r")
                f_tmp_csv=open(OAI_data_sheet+"."+MPI.Get_processor_name(),"a")
                for line in f_parsed_csv:
                    if list_folder_1[m] in line and folder_1 in line:
                        f_tmp_csv.write(line)
                        set_m.add(list_folder_1[m])
                        if(list_folder_1_flag[m] is "no"):
                            list_folder_1_flag[m]="yes"
                f_parsed_csv.close()
                f_tmp_csv.close()

    set_l=set()
    for l in range(stride_2,((comm.rank*folder_2_partition)+folder_2_partition)):
        f_parsed_csv=open("./parsed_csv.csv","r")
        f_tmp_csv=open(OAI_data_sheet+"."+MPI.Get_processor_name(),"a")
        for line in f_parsed_csv:
            if list_folder_2[l] in line and folder_2 in line:
                f_tmp_csv.write(line)
                set_l.add(list_folder_2[l])
                if(list_folder_2_flag[l] is "no"):
                    list_folder_2_flag[l]="yes"
        f_parsed_csv.close()
        f_tmp_csv.close()

    set_n=set()
    if folder_2_remaining > 0:
        for n in range((((comm.size-1)*folder_2_partition)+folder_2_partition),len(list_folder_2)):
            if ((len(list_folder_2))%n)==comm.rank:
                f_parsed_csv=open("./parsed_csv.csv","r")
                f_tmp_csv=open(OAI_data_sheet+"."+MPI.Get_processor_name(),"a")
                for line in f_parsed_csv:
                    if list_folder_2[n] in line and folder_2 in line:
                        f_tmp_csv.write(line)
                        set_n.add(list_folder_2[n])
                        if(list_folder_2_flag[n] is "no"):
                            f_second.write(list_folder_2[n]+"\n")
                            list_folder_2_flag[n]="yes"
                f_parsed_csv.close()
                f_tmp_csv.close()
    comm.Barrier()


if __name__ == '__main__':
    use_nifti=False
    avsm_path = "/home/uray/new_OAI/easyreg"
    rand_id = int(random.random()*10000)
    parse_csv()
    create_csv()
    #demo_analyze_single_image(use_nifti=use_nifti,avsm_path=avsm_path,do_clean=True)
    demo_analyze_cohort(use_nifti=use_nifti,avsm_path=avsm_path)
