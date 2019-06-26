"""
Registration using NiftyReg which is a command line registration tool

Author: Zhenlin Xu
Date；09/10/2018
"""

import subprocess
import os
from abc import ABC, abstractmethod
import sys
sys.path.append(os.path.realpath('..'))
import torch
from misc.str_ops import replace_extension
import shutil

class Register(ABC):
    @abstractmethod
    def register_affine(self, *args, **kwargs):
        pass

    @abstractmethod
    def register_bspline(self, *args, **kwargs):
        pass

    @abstractmethod
    def warp_image(self, *args, **kwargs):
        pass

    @abstractmethod
    def warp_points(self, *args, **kwargs):
        pass

    # @abstractmethod
    # def inverse_affine(self, *args, **kwargs):
    #     pass
    #
    # @abstractmethod
    # def inverse_nonrigid(self, *args, **kwargs):
    #     pass



class AVSMReg(Register):
    def __init__(self,avsm_path=None,python_executable='python'):
        super(AVSMReg,self).__init__()
        self.avsm_path = avsm_path
        self.python_executable = python_executable
        cur_dir = os.getcwd()
        self.refering_task_path= os.path.join(cur_dir, 'settings/avsm')
        self.mermaid_setting_path=os.path.join(cur_dir, 'settings/avsm/mermaid_setting.json')

    def register_image(self,target_path, moving_path,lmoving_path=None, ltarget_path=None,gpu_id=0,oai_image=None):
        output_path = os.path.join(os.path.split(oai_image.inv_transform_to_atlas)[0],'detailed')
        cmd = ''
        cmd +='{} single_pair_atlas_registration.py -rt {} \
        -s  {}  -t {}  -ls {}  -lt {}\
        -ms {}\
        -o {}\
         -g {}'.format(self.python_executable, self.refering_task_path,moving_path,target_path,lmoving_path,ltarget_path,self.mermaid_setting_path,output_path,gpu_id)
        wd = os.getcwd()
        #subprocess.run('source activate torch4 && {} && source deactivate'.format(cmd),cwd=self.avsm_path, shell=True)
        process = subprocess.Popen(cmd, cwd=self.avsm_path,shell=True)
        process.wait()
        os.chdir(wd)
        os.rename(os.path.join(output_path,'reg/res/records/original_sz/image_preprocessed_atlas_inv_phi.nii.gz'),oai_image.inv_transform_to_atlas)
        shutil.rmtree(output_path)
    def register_affine(self, *args, **kwargs):
        pass

    def register_bspline(self, *args, **kwargs):
        pass

    def warp_image(self, *args, **kwargs):
        pass

    def _get_inv_transfrom_map(self):
        pass

    def invert_nonrigid(self, non_rigid_transform, moving_image, inverted_transform, reference_image=None,
                        num_proc=None):
        pass

    def warp_points(self,points, inv_map,ref_img=None):
        """
        in avsm the inv transform coord is from [0,1], so here we need to read mesh in voxel coord and then normalized it to [0,1],
        the last step is to transform warped mesh into word coord/ voxel coord
        the transfom map use default [0,1] coord unless the ref img is provided
        here the transform map can be in inversed (height, width, depth) voxel space or in  inversed physical space (height, width, depth)
        but the points should be in standard voxel space (depth, height, width)
        :return:
        """

        import numpy as np
        import SimpleITK as sitk
        import torch.nn.functional as F
        # first make everything in voxel coordinate, depth, height, width
        img_sz=  np.array(inv_map.shape[1:])
        standard_spacing = 1/(img_sz-1)  # height, width, depth
        standard_spacing = np.flipud(standard_spacing) # depth, height, width
        img = sitk.ReadImage(ref_img)
        spacing = img.GetSpacing()
        spacing = np.array(spacing)

        points = points/spacing*standard_spacing
        points = points*2-1
        grid_sz =[1]+ [points.shape[0]]+ [1,1,3] # 1*N*1*1*3
        grid = points.reshape(*grid_sz)
        grid = torch.Tensor(grid).cuda()
        inv_map_sz = [1,3]+list(img_sz)  # height, width, depth
        inv_map = inv_map.view(*inv_map_sz) # 1*3*X*Y*Z
        points_wraped=F.grid_sample(inv_map, grid, mode='bilinear', padding_mode='border') # 1*3*N*1*1
        points_wraped = points_wraped.detach().cpu().numpy()
        points_wraped = np.transpose(np.squeeze(points_wraped))
        points_wraped = np.flip(points_wraped,1)/standard_spacing*spacing
        return points_wraped


    def warp_mesh(self, mesh_file, inv_transform_file, reference_image_file, warped_mesh_file=None, inWorld=False, do_clean=False):
        """
        warp a surface mesh with given transform (affine, bspline et al.)

        :param mesh_file: mesh to be warped, should be in voxel coord
        :param inv_transform_file: the inverse transform of registration, if inverse transform is given, invertT should be False
        :param reference_image: the image where the transformation is defined on, since we used inverse transformation
                to warp images, it should be the floating image of registration
        :param inWorld: if the mesh lies in the world coordinates.
                    If false, the mesh has to be transformed from voxel coordinates into world coordinates
        :param remove: if remove the intermediate points files when finished
        :return:
        """
        import nibabel as nib
        import shape_analysis.cartilage_shape_processing as csp
        inv_map = nib.load(inv_transform_file)
        inv_map = inv_map.get_fdata()
        inv_map = torch.Tensor(inv_map).cuda()
        points = csp.get_points_from_mesh(mesh_file).copy()
        transform = csp.get_voxel_to_world_transform_nifti(reference_image_file)
        points = csp.word_to_voxel_coord(points,transform)
        warped_points = self.warp_points(points, inv_map, reference_image_file)
        warped_points = csp.voxel_to_world_coord(warped_points,transform)
        csp.modify_mesh_with_new_vertices(mesh_file, warped_points, warped_mesh_file)
        if do_clean:
            os.remove(inv_transform_file)
        return warped_mesh_file



class NiftyReg(Register):
    def __init__(self, niftyreg_bin_path=None):
        super(NiftyReg, self).__init__()
        self.niftyreg_bin_path = niftyreg_bin_path

    def register_affine(self, ref_image_path: str, moving_image_path: str,
                        out_affine_file: str = None,
                        warped_image_path: str = None,
                        init_affine_file: str = None,
                        max_iterations: int = None,
                        num_levels: int = None,
                        performed_levels: int = None,
                        smooth_ref: int = None,
                        smooth_moving: int = None,
                        pv=None,
                        pi=None,
                        platf=0,
                        gpu_id=None,
                        num_threads=32,
                        **kwargs
                        ):
        """

        :param ref_image_path: reference image file
        :param moving_image_path:
        :param warped_image_path:
        :param init_affine_file:
        :param out_affine_file:
        :param max_iterations:
        :param num_levels:
        :param performed_levels:
        :param smooth_ref:
        :param smooth_moving:
        :param pv:
        :param pi:
        :param platf:
        :param gpu_id:
        :param num_threads:
        :param kwargs:
        :return:
        """
        if not os.path.isdir(os.path.dirname(out_affine_file)):
            os.makedirs(os.path.dirname(out_affine_file))

        cmd = '{}reg_aladin {} {}'.format(
            (self.niftyreg_bin_path + '/') if self.niftyreg_bin_path else '',
            ' '.join(['-ref {}'.format(ref_image_path),
                      '-flo {}'.format(moving_image_path),
                      '-res {}'.format(warped_image_path),
                      '-aff {}'.format(out_affine_file) if out_affine_file else '',
                      '-inaff {}'.format(init_affine_file) if init_affine_file else '',
                      '-maxit {}'.format(max_iterations) if max_iterations else '',

                      '-ln {}'.format(num_levels) if num_levels else '',
                      '-lp {}'.format(performed_levels) if performed_levels else '',

                      '-pv {}'.format(pv) if pv else '',
                      '-pi {}'.format(pi) if pi else '',
                      '-smooR {}'.format(smooth_ref) if smooth_ref else '',
                      '-smooF {}'.format(smooth_moving) if smooth_moving else '',
                      '-platf {}'.format(platf) if platf else '',
                      '-gpuid {}'.format(gpu_id) if gpu_id and platf == 1 else '',
                      '-omp {}'.format(num_threads) if num_threads and platf == 0 else '',
                      ]),
            # ' '.join(['-{} {}'.format(key, '' if (type(value) == type(True)) else value)
            #           for key, value in kwargs.items()])
            ' '.join(filter(lambda x: x is not '', [self.parse_args(key, value) for key, value in kwargs.items()]))
        )
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def register_bspline(self, ref_image_path, moving_image_path,
                         warped_image_path=None,
                         init_affine_file=None,
                         output_control_point=None,
                         max_iterations=None,
                         num_levels=None,
                         performed_levels=None,
                         smooth_ref=None,
                         smooth_moving=None,
                         num_threads=8,
                         platf=0,
                         gpu_id=None,
                         **kwargs
                         ):

        if not os.path.isdir(os.path.dirname(os.path.realpath(output_control_point))):
            os.makedirs(os.path.dirname(os.path.realpath(output_control_point)))

        cmd = '{}reg_f3d {} {}'.format(
            (self.niftyreg_bin_path + '/') if self.niftyreg_bin_path else '',
            ' '.join(['-ref {}'.format(ref_image_path),
                      '-flo {}'.format(moving_image_path),
                      '-res {}'.format(warped_image_path),
                      '-aff {}'.format(init_affine_file) if init_affine_file else '',
                      '-cpp {}'.format(output_control_point) if output_control_point else '',
                      '-maxit {}'.format(max_iterations) if max_iterations else '',
                      '-ln {}'.format(num_levels) if num_levels else '',
                      '-lp {}'.format(performed_levels) if performed_levels else '',
                      '-smooR {}'.format(smooth_ref) if smooth_ref else '',
                      '-smooF {}'.format(smooth_moving) if smooth_moving else '',
                      '-platf {}'.format(platf) if platf else '',
                      '-gpuid {}'.format(gpu_id) if gpu_id and platf == 1 else '',
                      '-omp {}'.format(num_threads) if num_threads and platf == 0 else '',
                      ]),
            # ' '.join([self.parse_args(key, value) for key, value in kwargs.items()])
            ' '.join(filter(lambda x: x is not '', [self.parse_args(key, value) for key, value in kwargs.items()]))
        )

        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def transform_to_deformation_field(self, transform_file, deformation_file, ref_image=None, num_proc=None):
        """
        Transform any kind of transformation into deformation field.
        The Reference image has to be specified when a cubic B-Spline parametrised control point grid is used
        :param transform_file: Input transformation file name
        :param deformation_file: Output deformation field file name
        :param ref_image: reference image file name
        :param num_proc: Number of thread to use with OpenMP, default to used all available ones
        :return:
        """
        self.__reg_transform(deform=(transform_file, deformation_file), ref=ref_image, omp=num_proc)

    def warp_points(self, transform_file, ref_image, input_points_file, output_landmark_file, num_proc=None):
        """
        Apply a transformation to a set of points(s).
        Landmarks are encoded in a text file with one point coordinates (mm) per line:
                <key1_x> <key1_y> <key1_z>
                <key2_x> <key2_y> <key2_z>

        :param transform_file: Input transformation file name
        :param input_points_file: Input landmark file name.
        :param output_landmark_file: Output landmark file name
        :param num_proc: Number of thread to use with OpenMP, default to used all available ones
        :return:
        """
        self.__reg_transform(land=(transform_file, input_points_file, output_landmark_file), ref=ref_image, omp=num_proc)

    def warp_mesh(self, mesh_file, inv_transform_file, reference_image_file, warped_mesh_file=None, inWorld=False, remove=True, do_clean=False):
        """
        warp a surface mesh with given transform (affine, bspline et al.)

        :param mesh_file: mesh to be warped
        :param inv_transform_file: the inverse transform of registration, if inverse transform is given, invertT should be False
        :param reference_image: the image where the transformation is defined on, since we used inverse transformation
                to warp images, it should be the floating image of registration
        :param inWorld: if the mesh lies in the world coordinates.
                    If false, the mesh has to be transformed from voxel coordinates into world coordinates
        :param remove: if remove the intermediate points files when finished
        :return:
        """
        # set filenames
        from shape_analysis.cartilage_shape_processing import mesh_to_nifti_world_coord, modify_mesh_with_new_vertices
        mesh_points_file = replace_extension(mesh_file, '_points.txt')
        warped_points_file = replace_extension(mesh_file, '_points_warped_with_{}.txt'.format(inv_transform_file.split(
            '/')[-1].split('.')[0]))

        # get vertices coordinates in the world space
        mesh_to_nifti_world_coord(mesh_file, reference_image_file, mesh_points_file, inWorld)

        # warped points with inverse transform
        self.warp_points(transform_file=inv_transform_file, ref_image=reference_image_file,
                         input_points_file=mesh_points_file, output_landmark_file=warped_points_file)

        # get warped mesh with warped points
        modify_mesh_with_new_vertices(mesh_file, warped_points_file, warped_mesh_file)

        if remove:
            os.remove(mesh_points_file)
            os.remove(warped_points_file)

        return warped_mesh_file

    def invert_affine(self, affine_transform, inverted_transform, num_proc=None):
        """
        Invert an affine matrix.

        :param affine_transform: Input affine transformation file name
        :param inverted_transform: Output inverted affine transformation file name
        :param num_proc: Number of thread to use with OpenMP, default to used all available ones
        :return: 
        """
        self.__reg_transform(invAff=(affine_transform, inverted_transform), omp=num_proc)

    def invert_nonrigid(self, non_rigid_transform, moving_image, inverted_transform, reference_image=None, num_proc=None):
        """
        Invert a non-rigid transformation and save the result as a deformation field.
        Note that the cubic b-spline grid parametrisations can not be inverted without approximation,
        as a result, they are converted into deformation fields before inversion.

        :param non_rigid_transform: Input transformation file name
        :param reference_image(optional): required when a cubic B-Spline parametrised control point grid is used
        :param moving_image: Input floating image where the inverted transformation is defined
        :param inverted_transform: Output inverted transformation file name
        :param num_proc: Number of thread to use with OpenMP, default to used all available ones
        :return:
        """
        if reference_image:
            self.__reg_transform(invNrr=(non_rigid_transform, moving_image, inverted_transform), ref=reference_image, omp=num_proc)
        else:
            self.__reg_transform(invNrr=(non_rigid_transform, moving_image, inverted_transform), omp=num_proc)

    def composite_transforms(self, transform1, ref_image1, transform2, ref_image2, output_transform):
        """
        Compose two transformations of any recognised type* and returns a deformation field.
                Trans3(x) = Trans2(Trans1(x)).

        :param ref_image1: where
        :param ref_image2:
        :param transform1: Input transformation 1 file name (associated with -ref if required)
        :param transform2: Input transformation 2 file name (associated with -ref2 if required)
        :param output_transform: Output deformation field file name
        :return:
        """

    def warp_image(self, ref_image, moving_image, transform_file, warped_image_file, warped_grid_file=None,
                   interp_order=3, padding=0, tensor=False, num_proc=None):
        """

        :param ref_image: Filename of the reference image (mandatory)
        :param moving_image: Filename of the floating image (mandatory)
        :param transform_file: Filename of the file containing the transformation parametrisation
        :param warped_image_file: Filename of the resampled image
        :param warped_grid_file: Filename of the resampled blank grid
        :param interp_order: Interpolation order (0, 1, 3, 4) (0=NN, 1=LIN; 3=CUB, 4=SINC) default:3
        :param padding: Interpolation padding value default:3
        :param tensor: The last six timepoints of the floating image are considered to be tensor order as XX, XY, YY, XZ, YZ, ZZ
        :param num_proc: Number of thread to use with OpenMP, default to used all available ones
        :return:
        """
        self.__reg_resample(ref=ref_image, flo=moving_image, trans=transform_file, res=warped_image_file,
                            blank=warped_grid_file, inter=interp_order, pad=padding, tensor=tensor, omp=num_proc)

    def average_images(self, output, images:list):
        """
        get average images
        :param images: input images
        :param output: output average image
        :return:
        """
        self.__reg_average(output, avg=images)

    def __reg_transform(self, **kwargs):
        """Interface to call reg_transform command tool"""

        cmd = '{}reg_transform {}'.format(
            (self.niftyreg_bin_path + '/') if self.niftyreg_bin_path else '',
            ' '.join(filter(lambda x: x is not '', [self.parse_args(key, value) for key, value in kwargs.items()]))
        )
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def __reg_resample(self, **kwargs):
        """Interface to call reg_resample command tool"""
        cmd = '{}reg_resample {}'.format(
            (self.niftyreg_bin_path + '/') if self.niftyreg_bin_path else '',
            ' '.join(filter(lambda x: x is not '', [self.parse_args(key, value) for key, value in kwargs.items()]))
        )
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def __reg_average(self, output, **kwargs):
        """Interface to call reg_resample command tool"""
        cmd = '{}reg_average {} {}'.format(
            (self.niftyreg_bin_path + '/') if self.niftyreg_bin_path else '',
            output,
            ' '.join(filter(lambda x: x is not '', [self.parse_args(key, value) for key, value in kwargs.items()]))
        )
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    @staticmethod
    def parse_args(key, value):
        """
        function to parse a python function kwargs pair as args of NiftiReg commands
        :param key:
        :param value:
        :return:
        """
        if value is not None:
            if key == 'deform':
                key = 'def'

            if (key in ['nmi', 'rbn', 'fbn', 'lncc', 'mind', 'mindssc'] and (not isinstance(value, tuple))) or \
                    (key in ['ssd', 'ssdn', 'kld'] and (isinstance(value, bool))):
                prefix = '--'
            else:
                prefix = '-'

            if isinstance(value, bool):
                if value:
                    return "{}{}".format(prefix, key)
                else:
                    return ''
            elif isinstance(value, tuple) or isinstance(value, list):
                return "{}{} {}".format(prefix, key, ' '.join(elem for elem in value))
            else:
                return "{}{} {}".format(prefix, key, value)
        else:
            return ''

    def __call__(self, moving_image, target_image, affine_config=None, nonrigid_config=None,):
        pass

    def register_image_to_atlas(self, image, config, atlas_image_path, moving_image_path, output_inv_map_path):
        pass

def demo_niftyreg():
    test_name = 'nifty_reg_lncc'
    moving_image_path = '/playpen-raid/zhenlinx/Data/OAI/9010952/MR_SAG_3D_DESS/LEFT\ KNEE/36\ MONTH/image_normalized.nii.gz'
    target_image_path = '/playpen-raid/zhenlinx/Data/OAI/9010952/MR_SAG_3D_DESS/LEFT\ KNEE/ENROLLMENT/image_normalized.nii.gz'

    # moving_image_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20041118_SAG_3D_DESS_LEFT_016610296205_image.nii.gz'
    # moving_label_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20041118_SAG_3D_DESS_LEFT_016610296205_label_all.nii.gz'
    # target_image_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_image.nii.gz'
    # target_label_path = '/playpen/zhenlinx/Data/OAI_segmentation/Nifti_rescaled/9003406_20060322_SAG_3D_DESS_LEFT_016610899303_label_all.nii.gz'

    if not os.path.exists('test_data/{}'.format(test_name)):
        os.mkdir('test_data/{}'.format(test_name))

    warped_affine_image_path = 'test_data/{}/moving_image_warpped_affine.nii.gz'.format(test_name)
    warped_bspline_image_path = 'test_data/{}/moving_image_warpped_bspline.nii.gz'.format(test_name)

    out_affine_file = 'test_data/{}/output_affine.txt'.format(test_name)
    reg = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")

    # affine registration
    reg.register_affine(target_image_path, moving_image_path, warped_affine_image_path, out_affine_file=out_affine_file,
                        # max_iterations=30,
                        platf=1, gpu_id=1, omp=32,
                        smooth_moving=-3, smooth_ref=-3,
                        pv=50, pi=50)


if __name__ == '__main__':
    # demo_niftyreg()


    pass
    # affine_T = 'test_analysis/9007827_20041006_SAG_3D_DESS_LEFT_016610263603_affine_transform.txt'
    # reg = NiftyReg("/playpen/zhenlinx/Code/niftyreg/install/bin")
    # reg.warp_points(affine_T, ref_image, 'test_analysis/points.txt', 'test_analysis/points_affine.txt')
