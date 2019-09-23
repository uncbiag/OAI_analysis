#!/usr/bin/env python
"""
Created by zhenlinx on 9/19/19
"""
import SimpleITK as sitk

import visvis as vv
import pymesh


# prob = sitk.ReadImage('/data/raid1/oai_analysis_results/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/TC_probmap.nii.gz')
# mask = prob > 0.5
# sitk.WriteImage(mask, '/data/raid1/oai_analysis_results/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/TC_mask.nii.gz')

FC_mesh_world = pymesh.load_mesh('/playpen/zhenlinx/Data/OAI_image_analysis/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/FC_mesh_world.ply')
TC_mesh_world = pymesh.load_mesh('/playpen/zhenlinx/Data/OAI_image_analysis/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/TC_mesh_world.ply')
FC_mesh_atlas = pymesh.load_mesh('/playpen/zhenlinx/Data/OAI_image_analysis/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/atlas_FC_mesh_with_thickness.ply')
TC_mesh_atlas = pymesh.load_mesh('/playpen/zhenlinx/Data/OAI_image_analysis/9000099/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/atlas_TC_mesh_with_thickness.ply')

app = vv.use()
a11 = vv.subplot(321)
FC_vis = vv.mesh((FC_mesh_world.vertices), FC_mesh_world.faces, values=FC_mesh_world.get_attribute('vertex_thickness'))
TC_vis = vv.mesh((TC_mesh_world.vertices), TC_mesh_world.faces, values=TC_mesh_world.get_attribute('vertex_thickness'))
FC_vis.colormap = vv.CM_JET
TC_vis.colormap = vv.CM_JET
vv.colorbar()

a12 = vv.subplot(322)
FC_vis = vv.mesh((FC_mesh_atlas.vertices), FC_mesh_atlas.faces, values=FC_mesh_atlas.get_attribute('vertex_thickness'))
TC_vis = vv.mesh((TC_mesh_atlas.vertices), TC_mesh_atlas.faces, values=TC_mesh_atlas.get_attribute('vertex_thickness'))
FC_vis.colormap = vv.CM_JET
TC_vis.colormap = vv.CM_JET
vv.colorbar()

a21 = vv.subplot(323)
FC_vis_copy = vv.mesh((FC_mesh_world.vertices), FC_mesh_world.faces,
                      values=FC_mesh_world.get_attribute('vertex_thickness'))
FC_vis_copy.colormap = vv.CM_JET
vv.colorbar()
a22 = vv.subplot(324)
FC_vis_copy_atlas = vv.mesh((FC_mesh_atlas.vertices), FC_mesh_atlas.faces, values=FC_mesh_atlas.get_attribute('vertex_thickness'))

FC_vis_copy_atlas.colormap = vv.CM_JET
vv.colorbar()

a31 = vv.subplot(325)
TC_vis_copy_atlas = vv.mesh((TC_mesh_world.vertices), TC_mesh_world.faces,
                      values=TC_mesh_world.get_attribute('vertex_thickness'))
TC_vis_copy_atlas.colormap = vv.CM_JET
vv.colorbar()
a32 = vv.subplot(326)
TC_vis_copy_atlas = vv.mesh((TC_mesh_atlas.vertices), TC_mesh_atlas.faces, values=TC_mesh_atlas.get_attribute('vertex_thickness'))
TC_vis_copy_atlas.colormap = vv.CM_JET
vv.colorbar()



app.Run()
