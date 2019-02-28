import nrrd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation,binary_erosion,interpolation
from scipy.misc import imresize
import subprocess,shutil,os
import SimpleITK as sitk
temp_dir='mip_temp'
from os.path import join
import os,glob  #for clear_dir(directory)


si = subprocess.STARTUPINFO()
si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
ffmpeg="C:\\Price\\ffmpeg-20180422-9f9f56e-win64-static\\bin\\ffmpeg.exe"

def clear_dir(directory):
    files=glob.glob(join(directory,'*'))
    for f in files:
        os.remove(f)

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


def rotate_and_flatten(im,degree):
    ar=sitk.GetArrayFromImage(im)
    ar=np.swapaxes(ar,0,2)
    rotated=interpolation.rotate(ar,degree,reshape=False,order=0,mode='constant',cval=0,prefilter=False)
    flattened=np.amax(rotated,axis=1)
    flattened=np.swapaxes(flattened,0,1)
    flattened=np.flipud(flattened)
    return flattened


def rotate_and_flatten_label(label,degree):
    ar=sitk.GetArrayFromImage(label)
    ar=np.swapaxes(ar,0,2)
    rotated=interpolation.rotate(ar,degree,reshape=False,order=0,mode='constant',cval=0,prefilter=False)
    flattened=np.amax(rotated,axis=1)
    flattened=np.swapaxes(flattened,0,1)
    flattened=np.flipud(flattened)
    dilated=binary_dilation(flattened,iterations=1)
    edge=dilated-flattened
    return edge

#im=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\PT1_15.nrrd")
#flat=rotate_and_flatten(im,0)
#plt.imshow(flat)
#plt.show()



def create_gif(im1,im2,im3,label,crop_distance,output_path,angle=20,z_scale=1.0,slowmo_factor=5,figure_size=[8,8]):
    print('Generating gif:',output_path)
    clear_dir(temp_dir)
    #if os.path.exists(temp_dir):
    ##    try:
    #       shutil.rmtree(temp_dir)
    #    except:
    #        print 'temp locked'
    #try:
    #    os.mkdir(temp_dir)
    #except:
    #    print('dir exists')
    current_deg=0
    f=plt.figure(figsize=figure_size)
    im1,label_crop,label_expanded_crop=crop_to_label(im1,label,crop_distance,True)
    im2,label_crop,label_expanded_crop=crop_to_label(im2,label,crop_distance,True)
    if im3:
        im3,label_crop,label_expanded_crop=crop_to_label(im3,label,crop_distance,True)
    ar1=sitk.GetArrayFromImage(im1)
    #ar2=sitk.GetArrayFromImage(im2)
    #if im3:
    #    ar3=sitk.GetArrayFromImage(im3)
    max_val=np.percentile(ar1,99.9)
    image_counter=1
    while current_deg<360:
        plt.clf()
        f1=rotate_and_flatten(im1,current_deg)
        f1=255*f1/f1.max()
        f2=rotate_and_flatten(im2,current_deg)
        f2=255*f2/f2.max()
        if im3:
            f3=rotate_and_flatten(im3,current_deg)
            f3=255*f3/f3.max()
        edge=rotate_and_flatten_label(label_crop,current_deg)
        edge=np.ma.masked_where(edge==0,edge)
        rgb=np.zeros((f1.shape[0],f1.shape[1],3))
        rgb[...,0]=f1
        rgb[...,1]=f2
        if im3:
            rgb[...,2]=f3
        #rgb=rgb*255/rgb.max()
        reverse=True
        if reverse:
            rgb=255-rgb
        plt.imshow(rgb.astype('uint8'),interpolation='nearest')
        plt.imshow(edge,interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0,hspace=0,top=1.0,left=0.0,right=1.0)
        plt.savefig(join(temp_dir,str(image_counter).zfill(4)+'.png'))
        #print(current_deg)
        current_deg+=angle
        image_counter+=1
        
        #ar1_r=interpolation.rotate(ar1,current_deg,reshape=False,order=0,mode='constant',cval=0.0,prefilter=False)
        #flat1=np.flipud(np.amax(ar1_r,axis=0).swapaxes(0,1))
        #flat1_s=imresize(flat1,[flat1.shape[0],int(flat1.shape[1].z_scale)])
    call=ffmpeg+' -y -i '+join(temp_dir,'%04d.png')+' -filter:v "setpts='+str(slowmo_factor)+'*PTS" '+output_path
    print(subprocess.Popen(call, stdout=subprocess.PIPE,startupinfo=si).stdout.read())
       
    
"""    
im1=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\PT1_15.nrrd")
im2=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\tfx_test\\result.nrrd")
im3=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\tfx_test2\\result.nrrd")
label=sitk.ReadImage("E:\\LuPSMA_SingleTimepoint\\BELLCYRIL\\structs\\xLVert.mha")
crop_distance=75
output_path='mip_temp\\test.gif'

create_gif(im1,im2,im3,label,crop_distance,output_path,angle=5,z_scale=1.0,slowmo_factor=2,figure_size=[10,10])
"""
#plt.imshow(flat)
#plt.show()

def create_mip(nrrd1_path, nrrd2_path, nrrd3_path, structure_name, output_path=False):
    if output_path==False:
        output_path=structure_name+'.jpg'

    nr1=nrrd.read(nrrd1_path)
    nr2=nrrd.read(nrrd2_path)
    if nrrd3_path:
        nr3=nrrd.read(nrrd3_path)

    #nr1=nrrd.read('burger_test_data\\cropped_pt.nrrd')
    #nr2=nrrd.read('burger_test_data\\cropped_pt2.nrrd')
    #nr3=nrrd.read('burger_test_data\\cropped_pt3.nrrd')
    #structure_name='test'
    #output_path='burger_test_data\\mips\\test.png'

    sum_axis=0
    f=plt.figure(figsize=[16,8])
    plt.subplot(131)

    ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    if nrrd3_path:
        ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=255-ar1
    im_ar[...,1]=255-ar2
    if nrrd3_path:
        im_ar[...,2]=255-ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)

    sum_axis=1

    ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    if nrrd3_path:
        ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=255-ar1
    im_ar[...,1]=255-ar2
    if nrrd3_path:
        im_ar[...,2]=255-ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)

    img1=Image.fromarray(im_ar.astype('uint8'),'RGB')

    #img1.show()

    sum_axis=2

    ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    if nrrd3_path:
        ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=255-ar1
    im_ar[...,1]=255-ar2
    if nrrd3_path:
        im_ar[...,2]=255-ar3
    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle(structure_name)
    plt.savefig(output_path)
    plt.close(f)
    return
    #plt.show()

    #img1=Image.fromarray(im_ar.astype('uint8'),'RGB')

    #img1.show()


def create_centre(nrrd1_path, nrrd2_path, nrrd3_path, structure_name, output_path=False):
    ct_min=-500
    ct_max=500
    if output_path==False:
        output_path=structure_name+'.jpg'

    nr1=nrrd.read(nrrd1_path)
    nr2=nrrd.read(nrrd2_path)
    nr3=nrrd.read(nrrd3_path)

    #nr1=nrrd.read('burger_test_data\\cropped_pt.nrrd')
    #nr2=nrrd.read('burger_test_data\\cropped_pt2.nrrd')
    #nr3=nrrd.read('burger_test_data\\cropped_pt3.nrrd')
    #structure_name='test'
    #output_path='burger_test_data\\mips\\test.png'

    sum_axis=0
    f=plt.figure(figsize=[16,8])
    plt.subplot(131)

    mid=int(nr1[0].shape[0]/2)
    
    #ar1=nr1[0][mid,:,:]*255/nr1[0][mid,:,:].max()
    #ar2=nr2[0][mid,:,:]*255/nr2[0][mid,:,:].max()
    #ar3=nr3[0][mid,:,:]*255/nr3[0][mid,:,:].max()

    ar1=nr1[0][mid,:,:]
    ar2=nr2[0][mid,:,:]
    ar3=nr3[0][mid,:,:]

    ar1=np.interp(ar1,(ct_min,ct_max),(0,255))
    ar2=np.interp(ar2,(ct_min,ct_max),(0,255))
    ar3=np.interp(ar3,(ct_min,ct_max),(0,255))
    


    #ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    #ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    #ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=ar1
    im_ar[...,1]=ar2
    im_ar[...,2]=ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(132)

    sum_axis=1

    #ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    #ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    #ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()

    mid=int(nr1[0].shape[1]/2)
    #ar1=nr1[0][:,mid,:]*255/nr1[0][:,mid,:].max()
    #ar2=nr2[0][:,mid,:]*255/nr2[0][:,mid,:].max()
    #ar3=nr3[0][:,mid,:]*255/nr3[0][:,mid,:].max()

    ar1=nr1[0][:,mid,:]
    ar2=nr2[0][:,mid,:]
    ar3=nr3[0][:,mid,:]

    ar1=np.interp(ar1,(ct_min,ct_max),(0,255))
    ar2=np.interp(ar2,(ct_min,ct_max),(0,255))
    ar3=np.interp(ar3,(ct_min,ct_max),(0,255))



    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=ar1
    im_ar[...,1]=ar2
    im_ar[...,2]=ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(133)

    img1=Image.fromarray(im_ar.astype('uint8'),'RGB')

    #img1.show()

    sum_axis=2

    #ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    #ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    #ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()

    mid=int(nr1[0].shape[2]/2)
    #ar1=nr1[0][:,:,mid]*255/nr1[0][:,:,mid].max()
    #ar2=nr2[0][:,:,mid]*255/nr2[0][:,:,mid].max()
    #ar3=nr3[0][:,:,mid]*255/nr3[0][:,:,mid].max()


    ar1=nr1[0][:,:,mid]
    ar2=nr2[0][:,:,mid]
    ar3=nr3[0][:,:,mid]

    ar1=np.interp(ar1,(ct_min,ct_max),(0,255))
    ar2=np.interp(ar2,(ct_min,ct_max),(0,255))
    ar3=np.interp(ar3,(ct_min,ct_max),(0,255))


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=ar1
    im_ar[...,1]=ar2
    im_ar[...,2]=ar3
    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle(structure_name)
    plt.savefig(output_path)
    plt.close(f)
    return

def create_both(ct1_path,ct2_path,ct3_path,pt1_path,pt2_path,pt3_path, structure_name, output_path=False):
    ct_min=-500
    ct_max=500
    if output_path==False:
        output_path=structure_name+'.jpg'

    nr1=nrrd.read(ct1_path)
    nr2=nrrd.read(ct2_path)
    nr3=nrrd.read(ct3_path)

    #nr1=nrrd.read('burger_test_data\\cropped_pt.nrrd')
    #nr2=nrrd.read('burger_test_data\\cropped_pt2.nrrd')
    #nr3=nrrd.read('burger_test_data\\cropped_pt3.nrrd')
    #structure_name='test'
    #output_path='burger_test_data\\mips\\test.png'

    sum_axis=0
    f=plt.figure(figsize=[16,16])
    plt.subplot(231)

    mid=int(nr1[0].shape[0]/2)
    
    #ar1=nr1[0][mid,:,:]*255/nr1[0][mid,:,:].max()
    #ar2=nr2[0][mid,:,:]*255/nr2[0][mid,:,:].max()
    #ar3=nr3[0][mid,:,:]*255/nr3[0][mid,:,:].max()

    ar1=nr1[0][mid,:,:]
    ar2=nr2[0][mid,:,:]
    ar3=nr3[0][mid,:,:]

    ar1=np.interp(ar1,(ct_min,ct_max),(0,255))
    ar2=np.interp(ar2,(ct_min,ct_max),(0,255))
    ar3=np.interp(ar3,(ct_min,ct_max),(0,255))
    


    #ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    #ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    #ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=ar1
    im_ar[...,1]=ar2
    im_ar[...,2]=ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(232)

    sum_axis=1

    #ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    #ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    #ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()

    mid=int(nr1[0].shape[1]/2)
    #ar1=nr1[0][:,mid,:]*255/nr1[0][:,mid,:].max()
    #ar2=nr2[0][:,mid,:]*255/nr2[0][:,mid,:].max()
    #ar3=nr3[0][:,mid,:]*255/nr3[0][:,mid,:].max()

    ar1=nr1[0][:,mid,:]
    ar2=nr2[0][:,mid,:]
    ar3=nr3[0][:,mid,:]

    ar1=np.interp(ar1,(ct_min,ct_max),(0,255))
    ar2=np.interp(ar2,(ct_min,ct_max),(0,255))
    ar3=np.interp(ar3,(ct_min,ct_max),(0,255))



    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=ar1
    im_ar[...,1]=ar2
    im_ar[...,2]=ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(233)

    img1=Image.fromarray(im_ar.astype('uint8'),'RGB')

    #img1.show()

    sum_axis=2

    #ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    #ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    #ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()

    mid=int(nr1[0].shape[2]/2)
    #ar1=nr1[0][:,:,mid]*255/nr1[0][:,:,mid].max()
    #ar2=nr2[0][:,:,mid]*255/nr2[0][:,:,mid].max()
    #ar3=nr3[0][:,:,mid]*255/nr3[0][:,:,mid].max()


    ar1=nr1[0][:,:,mid]
    ar2=nr2[0][:,:,mid]
    ar3=nr3[0][:,:,mid]

    ar1=np.interp(ar1,(ct_min,ct_max),(0,255))
    ar2=np.interp(ar2,(ct_min,ct_max),(0,255))
    ar3=np.interp(ar3,(ct_min,ct_max),(0,255))


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=ar1
    im_ar[...,1]=ar2
    im_ar[...,2]=ar3
    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')


    ###PET MIP SECTION

    nr1=nrrd.read(pt1_path)
    nr2=nrrd.read(pt2_path)
    nr3=nrrd.read(pt3_path)

    #nr1=nrrd.read('burger_test_data\\cropped_pt.nrrd')
    #nr2=nrrd.read('burger_test_data\\cropped_pt2.nrrd')
    #nr3=nrrd.read('burger_test_data\\cropped_pt3.nrrd')
    #structure_name='test'
    #output_path='burger_test_data\\mips\\test.png'

    sum_axis=0
    #f=plt.figure(figsize=[16,8])
    plt.subplot(234)

    ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=255-ar1
    im_ar[...,1]=255-ar2
    im_ar[...,2]=255-ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(235)

    sum_axis=1

    ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=255-ar1
    im_ar[...,1]=255-ar2
    im_ar[...,2]=255-ar3

    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')
    plt.subplot(236)

    img1=Image.fromarray(im_ar.astype('uint8'),'RGB')

    #img1.show()

    sum_axis=2

    ar1=nr1[0].max(axis=sum_axis)*255/nr1[0].max()
    ar2=nr2[0].max(axis=sum_axis)*255/nr2[0].max()
    ar3=nr3[0].max(axis=sum_axis)*255/nr3[0].max()


    im_ar=np.zeros((ar1.shape[0],ar1.shape[1],3))

    im_ar[...,0]=255-ar1
    im_ar[...,1]=255-ar2
    im_ar[...,2]=255-ar3
    plt.imshow(im_ar.astype('uint8'),interpolation='nearest')
    plt.axis('off')










    plt.tight_layout()
    plt.suptitle(structure_name)
    plt.savefig(output_path)
    plt.close(f)
    return
