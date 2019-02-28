
import shutil,os,subprocess
from os.path import join
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_dilation
from fused_mips_v2 import *



#subprocess flags
si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW


elastix='elastix'
transformix='transformix'
plastimatch='plastimatch'


working_dir='working_dir'

def new_dir(directory_path):
    try:
        os.mkdir(directory_path)
    except:
        #print(directory_path,'exists')
        zzz=1
    return

def run_reg(fixed_im,moving_im,params,out_folder,initial_params=False,label_mask=False):
    new_dir(out_folder)
    #if not initial_params:
    call=elastix+' -f '+fixed_im+' -m '+moving_im+' -p '+params+' -out '+out_folder
    #else:
    #    call=elastix+' -f '+fixed_im+' -m '+moving_im+' -p '+params+' -t0 '+initial_params+' -out '+out_folder
    if initial_params:
        call+=' -t0 '+initial_params
    if label_mask:
        call+=' -fMask '+label_mask
    print call
    output=subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read()
    if initial_params:
        f=open(initial_params,'r')
        init_lines=f.readlines()
        f.close()
        for l in init_lines:
            if '(TransformParameters' in l:
                p1=float(l.split(' ')[1])
                p2=float(l.split(' ')[2])
                p3=float(l.split(' ')[3].replace(')',''))
        f=open(join(out_folder,'TransformParameters.0.txt'),'r')
        reg_lines=f.readlines()
        f.close()
        out_lines=[]
        for l in reg_lines:
            if '(TransformParameters' in l:
                f1=float(l.split(' ')[1])
                f2=float(l.split(' ')[2])
                f3=float(l.split(' ')[3].replace(')',''))
                out_lines.append('(TransformParameters '+str((p1+f1))+' '+str((p2+f2))+' '+str((p3+f3))+')\n')
            elif 'InitialTransformParameters' in l:
                #print('SkippingLine')
                zzz=1
            else:
                out_lines.append(l)
        f=open(join(out_folder,'CombinedParams.txt'),'w')
        for l in out_lines:
            f.write(l)
        f.close()
    return output

def run_tfx(moving_im,params,out_folder):
    new_dir(out_folder)
    call=transformix+' -in '+moving_im+' -tp '+params+' -out '+out_folder
    output=subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read()
    return output

def overwrite_image(image,numpy_array,dtype='uint16'):
    newim=sitk.GetImageFromArray(numpy_array.astype(dtype))
    newim.SetOrigin(image.GetOrigin())
    newim.SetSpacing(image.GetSpacing())
    newim.SetDirection(image.GetDirection())
    return newim

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

def crop_to_label(sitk_im,sitk_label,dilation_range,resample_label=True):
    if resample_label:
        sitk_label=sitk.Resample(sitk_label,sitk_im,sitk.Transform(),sitk.sitkNearestNeighbor,0.)
    ar=sitk.GetArrayFromImage(sitk_label)
    ar_expanded=binary_dilation(ar,iterations=dilation_range)
    label_expanded=overwrite_image(sitk_label,ar_expanded)
    min_box,max_box=get_bounding_box(ar_expanded)
    crop=sitk.CropImageFilter()
    crop.SetUpperBoundaryCropSize(max_box)
    crop.SetLowerBoundaryCropSize(min_box)
    cropped_label=crop.Execute(sitk_label)
    cropped_expanded_label=crop.Execute(label_expanded)
    cropped_im=crop.Execute(sitk_im)
    return cropped_im,cropped_label, cropped_expanded_label

def dilate_label(fixed_im,label,dilation,resample_label=True,out_path=False):
    if type(label) is str:
        label=sitk.ReadImage(label)
    if type(fixed_im) is str:
        fixed_im=sitk.ReadImage(fixed_im)
    if resample_label:
        label=sitk.Resample(label,fixed_im,sitk.Transform(),sitk.sitkNearestNeighbor,0.)
    ar=sitk.GetArrayFromImage(label)
    ar_expanded=binary_dilation(ar,iterations=dilation)
    label_expanded=overwrite_image(label,ar_expanded)
    if out_path:
        sitk.WriteImage(label_expanded,out_path)

def params_from_file(transform_path):
    f=open(transform_path,'r')
    text=f.readlines()
    params=[]
    for line in text:
        if '(TransformParameters ' in line:
            params=line.replace('(TransformParameters ','').replace(')\n','').split(' ')
    return params
            

def multi_res_cropped(fixed_im,moving_im,label_path,dilation_list,initial_params,fine_params,out_dir):
    new_dir(out_dir)
    i=0
    #initial_params=False
    #params=[]
    for dilation in dilation_list:
        i+=1
        
        dilate_label(fixed_im,label_path,dilation,resample_label=True,out_path=join(working_dir,'dilated_label.nrrd'))
        label=sitk.ReadImage(join(working_dir,'dilated_label.nrrd'))
        if i>1:
        #if False:
            #new_dir(join(out_dir,'previous'))
            if i==2:
                shutil.copyfile(join(out_dir,'TransformParameters.0.txt'),join(working_dir,'t0.txt'))
            else:
                shutil.copyfile(join(out_dir,'CombinedParams.txt'),join(working_dir,'t0.txt'))
            initial_params=join(working_dir,'t0.txt') #out_dir
            #run_reg(fixed_im,moving_im,fine_params,out_dir,initial_params=join(working_dir,'t0.txt'),label_mask=join(working_dir,'dilated_label.nrrd'))
            output=run_reg(fixed_im,moving_im,fine_params,out_dir,initial_params=initial_params,label_mask=join(working_dir,'dilated_label.nrrd'))  #uncomment above to return sequential
            #(fixed_im,moving_im,params,out_folder,initial_params=False,label_mask=False):
        else:
            output=run_reg(fixed_im,moving_im,fine_params,out_dir,initial_params=initial_params,label_mask=join(working_dir,'dilated_label.nrrd'))
        #params.append(params_from_file(join(out_dir,'TransformParameters.0.txt')))
        params=params_from_file(join(out_dir,'TransformParameters.0.txt'))
        #im1=sitk.ReadImage(fixed_im)
        #im2=sitk.ReadImage(join(out_dir,'Result.0.nrrd'))
        #im3=False
        #label2=sitk.ReadImage(join(working_dir,'dilated_label.nrrd'))
        #crop_distance=60
        #output_path=join(working_dir,os.path.basename(label_path).split('.')[0]+'.gif')
        #create_gif(im1,im2,im3,label2,crop_distance,output_path,angle=30,z_scale=1.0,slowmo_factor=6,figure_size=[10,10])
        #shutil.copy('mip_temp\\test.gif',join(working_dir,str(dilation)+'.gif'))
    #print params
    return params,output
    
def dilated_fine_reg(fixed_im,moving_im,label_path,dilation,initial_params,fine_params,out_dir):
    new_dir(out_dir)
    dilate_label(fixed_im,label_path,dilation,resample_label=True,out_path=join(working_dir,'dilated_label.nrrd'))
    label=sitk.ReadImage(join(working_dir,'dilated_label.nrrd'))
    output=run_reg(fixed_im,moving_im,fine_params,out_dir,initial_params=initial_params,label_mask=join(working_dir,'dilated_label.nrrd'))
    return params_from_file(join(out_dir,'TransformParameters.0.txt')), output


fixed_ct='BELLCYRIL\\CT1_15.nrrd'
moving_ct='BELLCYRIL\\CT2.nrrd'
params='param\\rigid_param.txt'
out_folder='working_dir\\rigid2'
#run_reg(fixed_ct,moving_ct,params,out_folder)

initial_params='working_dir\\rigid2\\TransformParameters.0.txt'
fixed_pt='BELLCYRIL\\PT1_15.nrrd'
moving_pt='BELLCYRIL\\PT2.nrrd'
params='param\\rigid_param_focused_fast.txt'
out_folder='working_dir\\focused2'
#label='BELLCYRIL\\structs\\xLVert.mha'
label='BELLCYRIL\\structs\\Parotid_Lt.mha'
fine_params='param\\fine_params2.txt'
#dilation_list=[50,35,20,15,10,7,5,3,2,1]

#def dilate_2x_registration(fixed_im,moving_im1,

def print_output_vals(output):
    output=output.split('\n')
    for l in output:
        if 'final metric' in l.lower():
            print(l)
        if 'time spent in resolution' in l.lower():
            print(l)
    return

"""
dilation_list=[15]
for struct in os.listdir('BELLCYRIL\\structs'):
    label=join('BELLCYRIL\\structs',struct)
    multi_res_cropped(fixed_pt,moving_pt,label,dilation_list,initial_params,fine_params,out_dir='working_dir\\fine_multires')
#multi_res_cropped(fixed_pt,moving_pt,label,dilation_list,initial_params,fine_params,out_dir='working_dir\\fine_multires')
im1=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\PT1_15.nrrd")
im2=sitk.ReadImage('working_dir\\fine_multires\\result.0.nrrd')
im3=False
#label=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\structs\\xLVert.mha")
label=sitk.ReadImage(join(working_dir,'dilated_label.nrrd'))
crop_distance=75
output_path='mip_temp\\test.gif'
"""
                           

#create_gif(im1,im2,im3,label,crop_distance,output_path,angle=5,z_scale=1.0,slowmo_factor=2,figure_size=[10,10])

#run_reg(fixed_pt,moving_pt,params,out_folder,initial_params,label_mask=False)
        
