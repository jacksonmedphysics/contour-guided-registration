#!/usr/bin/python
import sys
import os
#sys.path.append('/home/price/dicom_writer')
import pydicom as dicom

#walks through a directory of dicom images
#return a list of the different series
#the order of that list points to a second list containing the ordered
#filenames for each series
#return [Series 1- Description, Series2, Series3,...]
#second list[[Series1 file1.dcm, series1 file2.dcm],[series2 file1.dcm,...]...]
def get_paths(dir_in):
    """finds the names of uniques series in a directory of dicom files
    for each unique series, it creates a sorted list of dicom files in a two-tier list
    returns [list, of, series, names] [[list, and, sublist],[of, sorted, paths]]
    """


    #directory='/home/price/Lu-PSMA_Dosimetry/Krywula_29-10-15/krywula'
    seriesdescriptions=[]
    paths=[]
    instances=[]
    seriesnums=[]
    acquisition_datetimes=[]
    series_uids=[]
    series_modalities=[]
    filelist=os.listdir(dir_in)
    num=1
    for (directory,_,files) in os.walk(dir_in):
        for name in files:
            path=os.path.join(directory,name)
            ext=path.split('.')[-1]
            if ext.lower()=='dcm' or ext.lower()=='ima':
                if num%50==0:
                    print 'Reading file ',num
                dcm=dicom.read_file(os.path.join(directory,name))
                seriesdescriptions.append(dcm.SeriesDescription)
                paths.append(os.path.join(directory,name))
                try:
                    instances.append(dcm.InstanceNumber)
                except:
                    instances.append(0)
                seriesnums.append(dcm.SeriesNumber)
                try:
                    acquisition_datetimes.append(str(dcm.AcquisitionDate)+' '+str(dcm.AcquisitionTime))
                except:
                    acquisition_datetimes.append('na')
                series_uids.append(dcm.SeriesInstanceUID)
                series_modalities.append(dcm.Modality)
                num+=1

    



                                
    series_paths_list=[]        
    series_names=[]


    unique_series_uids=[]
    i=0
    unique_datetimes=[]
    unique_modalities=[]
    for uid in series_uids:
        if uid not in unique_series_uids:
            unique_series_uids.append(uid)
            if seriesdescriptions[i] not in series_names:
                series_names.append(seriesdescriptions[i])
            else:
                series_names.append(seriesdescriptions[i]+str(i))
            unique_datetimes.append(acquisition_datetimes[i])
            unique_modalities.append(series_modalities[i])
        i+=1
            

    #for name in seriesnames:
    for uid in unique_series_uids:
        print uid
        series_instances=[]
        series_paths=[]
        for i in range(len(paths)):
            if series_uids[i]==uid:
                series_instances.append(instances[i])
                series_paths.append(paths[i])
                
        sorted_paths=[series_paths for (series_instances,series_paths) in sorted(zip(series_instances,series_paths))]
        series_paths_list.append(sorted_paths)
    return unique_series_uids, series_names, unique_datetimes, series_paths_list, unique_modalities
