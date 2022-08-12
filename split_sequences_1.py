import os
import os.path as osp
import cv2
import json 
import pandas as pd 
from itertools import groupby
from operator import itemgetter
def get_num_persons(det_results_file):
    with open(os.path.join(det_results_file),"r") as fdet:
        lines = fdet.readlines()
    person_ids = set()
    for line in lines:
        line = line.rstrip().split(",")
        
        id = int(line[1])
        person_ids.add(id)

    return len(person_ids)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="./batch_01/batch_01_p01", help="path to folder containing folders of each video"
    )
    args= parser.parse_args()
    
    with open(osp.join(args.data_dir,"annotations/video_names.txt") ,"r") as vid_names_f:
        lines = vid_names_f.readlines()

    data_dir = args.data_dir

    
    for line in lines:
        vid_name = line.rstrip()
        print(vid_name)
        tracking_results_file = os.path.join(data_dir,vid_name, f"tracking_results/{vid_name}.txt")
        n_persons = get_num_persons(tracking_results_file)
        
        anno_dict = {}

        anno_dict['n_persons'] = n_persons
        cap = cv2.VideoCapture(os.path.join(data_dir,vid_name,vid_name+".mp4"))
        anno_dict['n_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        

        anno_dict['sequences'] = {}
        
        tracking_data = pd.read_csv(tracking_results_file,header=None, sep=',',
                                    usecols=range(7),
                                    names=[ 'frame_id', 'person_id' ,'x1','y1','w','h', 'conf'])
      
        print("total tracked ids: ",tracking_data['person_id'].unique()) 
        for person_id, group in tracking_data.groupby('person_id'):
            print(f"==================== Person {person_id:02d} =====================")
            frame_list = sorted(list(group['frame_id']))
            
            sequence_anno = []
            print(f"person_{person_id} - tracked in {len(frame_list)} frames")
            for k, g in groupby(enumerate(frame_list), lambda i_x: i_x[0] - i_x[1]):
                l = list(map(itemgetter(1), g))
                start, end = l[0], l[-1]+ 1 #+1 since we want to loop range(start,end) instead of range(start,end+1) latter in the pipeline
                if end-start >= 10:
                    sequence_anno.append([start,end])
            print(sequence_anno)
            anno_dict['sequences'][f'person_{person_id:02d}'] = sequence_anno
        
        
    

        out_dir = osp.join(data_dir,'annotations', vid_name)
        os.makedirs(out_dir, exist_ok=True)

        with open(osp.join(out_dir, f"split_sequences.json"), "w") as f_out:
            json.dump(anno_dict, f_out)