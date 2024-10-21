import os
from glob import glob

import numpy as np


def main():
    contact_data_dict = {}
    
    
    ## collect data
    files = os.listdir("./dataset")
    for fname in files:
        fpath = os.path.join("./dataset", fname)
        
        with open(fpath, "r") as f:
            
            ## initial point
            first_line = f.readline()
            init_pt = [float(v.strip()) for v in first_line.split(",")[:3]]
            init_normal = [float(v.strip()) for v in first_line.split(",")[3:]]
            key_name = str(init_pt + init_normal)
            
            
            ## stack contact data for deviation
            for line in f.readlines():
                
                pt = [float(v) for v in line.split(",")[:3]]
                normal = [float(v) for v in line.split(",")[3:]]
                
                
                if key_name in contact_data_dict.keys():
                    contact_data_dict[key_name].append(pt + normal)
                else:
                    contact_data_dict[key_name] = [pt + normal]
            
            
            
    ## init_to_std_dict()
    init_to_std_dict = {}
    for k in contact_data_dict.keys():
        temp = np.array(contact_data_dict[k])
        print(k, temp.shape, temp.dtype)  ## (len, 6)
        
        
        means = np.mean(np.array(contact_data_dict[k]), axis=0)  ## num, 6
        std = np.mean(means)
        init_to_std_dict[k] = std
        
        
    ## save to file
    with open("std.txt", "w") as f:
        for k, v in init_to_std_dict.items():
            f.write(str(k) + "," + str(v) + "\n")
        
    
    



if __name__ == "__main__":
    main()