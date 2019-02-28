import SimpleITK as sitk
from os.path import join
import nrrd
import os
import subprocess
from scipy.ndimage import binary_dilation
#import matplotlib.pyplot as plt
import numpy as np
#import fit_func_triexp as ff
#from fused_mips import create_mip, create_centre, create_both
#import time

from series_lister import get_paths
import shutil
import datetime
import pydicom as dicom
import numpy

from fused_mips_v2 import *
from registration_sequences_v2 import *
#from plot_single import *


#note this expects each 

input_dir='input'
temp_dir='temp'

clear_temp=False
convert_dicom=False #True
resample=False #True
convert_rt=False #True
initial_rigid=False #True
wide_ct_expansion=40
pet_expansion=20 #old
crop_expansion=50 #previously 25
resolution_mm=1.5

pet_dilation=55 #old
ct_dilation=40
pet_dilation1=60
pet_dilation2=15


si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW







def new_dir(directory_path):
    try:
        os.mkdir(directory_path)
    except:
        #print(directory_path,'exists')
        zzz=1
    return

new_dir(temp_dir)
new_dir(join(temp_dir,'structs'))

#sort all series by datetime
if convert_dicom:
    unique_series_uids, series_names, unique_datetimes, series_paths_list, unique_modalities=get_paths(input_dir)
    zipped=zip(unique_datetimes, unique_series_uids, series_names, series_paths_list, unique_modalities)
    zipped.sort()
    unique_datetimes, unique_series_uids, series_names, series_paths_list, unique_modalities = zip(*zipped)
    reader=sitk.ImageSeriesReader()
    ct_iter=0
    pt_iter=0
    for i in range(len(series_names)):
        if unique_modalities[i].lower()=='ct':
            ct_iter+=1
            print('converting ', series_names[i],'as CT', str(ct_iter))
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(os.path.dirname(series_paths_list[i][0])))
            im=reader.Execute()
            sitk.WriteImage(im,join(temp_dir,'CT'+str(ct_iter)+'.nrrd'))
        if unique_modalities[i].lower()=='pt':
            pt_iter+=1
            print('converting ', series_names[i],'as PT', str(pt_iter))
            reader.SetFileNames(reader.GetGDCMSeriesFileNames(os.path.dirname(series_paths_list[i][0])))
            im=reader.Execute()
            sitk.WriteImage(im,join(temp_dir,'PT'+str(pt_iter)+'.nrrd'))
    for i in range(len(series_names)): #need to make sure CT1 is converted before plastimatch call to create rt structure set
        if unique_modalities[i].lower()=='rtstruct':
            print('Converting RT Structure Set')
            call='plastimatch convert --input '+series_paths_list[i][0]+' --output-prefix '+join(temp_dir,'structs')+' --fixed '+join(temp_dir,'CT1.nrrd')
            print(subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read())


if resample:
    print('Resampling CT')
    ct=sitk.ReadImage(join(temp_dir,'CT1.nrrd'),sitk.sitkInt16)
    sitk.WriteImage(ct,join(temp_dir,'CT1.nrrd'))
    res_string=str(resolution_mm)+' '+str(resolution_mm)+' '+str(resolution_mm)
    call='plastimatch resample --input '+join(temp_dir,'CT1.nrrd')+' --interpolation "linear" --spacing "'+res_string+'" --output '+join(temp_dir,'CT1_15.nrrd')
    print(subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read())
    call='plastimatch resample --input '+join(temp_dir,'PT1.nrrd')+' --interpolation "linear" --fixed '+join(temp_dir,'CT1_15.nrrd')+' --output '+join(temp_dir,'PT1_15.nrrd')
    print(subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read())

if initial_rigid:
    new_dir(join(temp_dir,'rigid_out'))
    call='elastix -f '+join(temp_dir,'CT1_15.nrrd')+' -m '+join(temp_dir,'CT2.nrrd')+' -p param\\rigid_param.txt -out '+join(temp_dir,'rigid_out')
    print(subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read())
    new_dir(join(temp_dir,'rigid_out2'))
    call='elastix -f '+join(temp_dir,'CT1_15.nrrd')+' -m '+join(temp_dir,'CT3.nrrd')+' -p param\\rigid_param.txt -out '+join(temp_dir,'rigid_out2')
    print(subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read())
    

struct_path=join(temp_dir,'structs')
struct_list=os.listdir(struct_path)
crop_moving=True

def overwrite_image(image,numpy_array,dtype='uint16'):
    newim=sitk.GetImageFromArray(numpy_array.astype(dtype))
    newim.SetOrigin(image.GetOrigin())
    newim.SetSpacing(image.GetSpacing())
    newim.SetDirection(image.GetDirection())
    return newim
                     

crop=sitk.CropImageFilter()

def get_bounding_box(numpy_array):
    line=np.sum(np.sum(numpy_array,axis=0),axis=0)
    xmin=np.where(line>0)[0].min()
    xmax=np.where(line>0)[0].max()
    line=np.sum(np.sum(numpy_array,axis=0),axis=-1)
    ymin=np.where(line>0)[0].min()
    ymax=np.where(line>0)[0].max()
    line=np.sum(np.sum(numpy_array,axis=-1),axis=-1)
    zmin=np.where(line>0)[0].min()
    zmax=np.where(line>0)[0].max() 
    #return [zmin,ymin,xmin],[zmax,ymax,xmax]
    return [int(xmin),int(ymin),int(zmin)],[int(numpy_array.shape[2]-xmax),int(numpy_array.shape[1]-ymax),int(numpy_array.shape[0]-zmax)]

new_dir(join(temp_dir,'mips'))
new_dir(join(temp_dir,'focused_out'))
new_dir(join(temp_dir,'focused_out2'))
new_dir(join(temp_dir,'tfx_out'))
new_dir(join(temp_dir,'tfx_out2'))
new_dir(join(temp_dir,'cropped_nrrds'))
print(struct_list)
pet_list=['subm','parot','lacrim'] #structures with these strings or starting with letter 'x' get focused registration by PET intensity, all others are registered by CT
r=open(join(temp_dir,'structure_measures.csv'),'w')
for struct in struct_list:
    single_structure_path=join(struct_path,struct)
    label=sitk.ReadImage(join(struct_path,struct))
    struct=struct.split('.')[0]
    if struct[0]=='x':
        pet_reg=True
    else:
        pet_reg=False
    if any(x in struct.lower() for x in pet_list):
        pet_reg=True
    if pet_reg:
        print('Corrected PET registration for', struct)
    ct=sitk.ReadImage(join(temp_dir,'CT1_15.nrrd'))
    pt=sitk.ReadImage(join(temp_dir,'PT1_15.nrrd'))
    label=sitk.Resample(label,ct,sitk.Transform(),sitk.sitkNearestNeighbor,0.)
    ar=sitk.GetArrayFromImage(label)
    sitk.WriteImage(label,join(temp_dir,'struct.nrrd'))
    line=struct+',,'
    if pet_reg:
        line+='PET,'
        #call='elastix -f '+join(temp_dir,'PT1_15.nrrd')+' -m '+join(temp_dir,'PT2.nrrd')+' -t0 '+join(temp_dir,'rigid_out','TransformParameters.0.txt')+' -p '+focused_params+' -threads 4 -out '+join(temp_dir,'focused_out')+' -fMask '+join(temp_dir,'cropped_expanded_label.nrrd')
        fixed_im=join(temp_dir,'PT1_15.nrrd')
        moving_im=join(temp_dir,'PT2.nrrd')
        label_path=join(struct_path,struct+'.mha')
        dilation=pet_dilation
        initial_params=join(temp_dir,'rigid_out','TransformParameters.0.txt')
        fine_params=join('param','fine_params2.txt')
        out_dir=join(temp_dir,'focused_out')
        #params,output=dilated_fine_reg(fixed_im,moving_im,label_path,dilation,initial_params,fine_params,out_dir)
        params,output=multi_res_cropped(fixed_im,moving_im,label_path,[pet_dilation1,pet_dilation2],initial_params,fine_params,out_dir)
        print_output_vals(output)
        line+=params[0]+','+params[1]+','+params[2]+',,'

        moving_im=join(temp_dir,'PT3.nrrd')
        initial_params=join(temp_dir,'rigid_out2','TransformParameters.0.txt')
        out_dir=join(temp_dir,'focused_out2')
        #params,output=dilated_fine_reg(fixed_im,moving_im,label_path,dilation,initial_params,fine_params,out_dir)
        params,output=multi_res_cropped(fixed_im,moving_im,label_path,[pet_dilation1,pet_dilation2],initial_params,fine_params,out_dir)
        print_output_vals(output)
        line+=params[0]+','+params[1]+','+params[2]+',,'

        pt1=sitk.ReadImage(join(temp_dir,'PT1_15.nrrd'))
        pt2=sitk.ReadImage(join(temp_dir,'focused_out','result.0.nrrd'))
        pt3=sitk.ReadImage(join(temp_dir,'focused_out2','result.0.nrrd'))
        #label exists already
        crop_distance=60
        output_path=join(temp_dir,'mips',struct+'.gif')
        create_gif(pt1,pt2,pt3,label,crop_distance,output_path)
    else:
        line+='CT,'
        fixed_im=join(temp_dir,'CT1_15.nrrd')
        moving_im=join(temp_dir,'CT2.nrrd')
        label_path=join(struct_path,struct+'.mha')
        dilation=ct_dilation
        initial_params=join(temp_dir,'rigid_out','TransformParameters.0.txt')
        fine_params=join('param','fine_params2.txt')
        out_dir=join(temp_dir,'focused_out')
        params,output=dilated_fine_reg(fixed_im,moving_im,label_path,dilation,initial_params,fine_params,out_dir)
        print_output_vals(output)
        line+=params[0]+','+params[1]+','+params[2]+',,'

        #Transformix
        moving_im=join(temp_dir,'PT2.nrrd')
        tfx_params=join(temp_dir,'focused_out','TransformParameters.0.txt')
        out_dir=join(temp_dir,'tfx_out')
        run_tfx(moving_im,tfx_params,out_dir)


        moving_im=join(temp_dir,'CT3.nrrd')
        initial_params=join(temp_dir,'rigid_out2','TransformParameters.0.txt')
        out_dir=join(temp_dir,'focused_out2')
        params,output=dilated_fine_reg(fixed_im,moving_im,label_path,dilation,initial_params,fine_params,out_dir)
        print_output_vals(output)
        line+=params[0]+','+params[1]+','+params[2]+',,'

        moving_im=join(temp_dir,'PT3.nrrd')
        tfx_params=join(temp_dir,'focused_out2','TransformParameters.0.txt')
        out_dir=join(temp_dir,'tfx_out2')
        run_tfx(moving_im,tfx_params,out_dir)

        

        pt1=sitk.ReadImage(join(temp_dir,'PT1_15.nrrd'))
        pt2=sitk.ReadImage(join(temp_dir,'tfx_out','result.nrrd'))
        pt3=sitk.ReadImage(join(temp_dir,'tfx_out2','result.nrrd'))
        #label exists already
        crop_distance=60
        output_path=join(temp_dir,'mips',struct+'.gif')
        create_gif(pt1,pt2,pt3,label,crop_distance,output_path)        
    
    p1=sitk.GetArrayFromImage(pt)
    p2=sitk.GetArrayFromImage(pt2)
    p3=sitk.GetArrayFromImage(pt3)
    values=[p1[ar==1].mean(),p2[ar==1].mean(),p3[ar==1].mean()]
    line+=str(values[0])+','+str(values[1])+','+str(values[2])+'\n'
    r.write(line)
r.close()
