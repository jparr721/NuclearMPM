import numpy as np
import pandas as pd
import pickle5 as pkl
from pathlib import Path

def parse_text_files(data_dir="../../build/tmp"):
    sim_dict_path = Path('sim_results_dict.pkl')
    if sim_dict_path.exists():
        print("Previous Simulation Results Found")
        with open(sim_dict_path, 'rb') as reader:
            dat = pkl.load(reader)
        return dat
    print("Fetching Simulation Results ",end="")

    data_dir = Path(data_dir)
    dat = {}
    params = 'x,v,F,C,Jp,lame,velocity,mass,timestep'
    params = params.split(',')
    resolution = 65 # TODO: read from file after simulation

    file_path_to_int = lambda path_str: int(path_str.parts[-1].split("_")[0])

    for var in params:
        # Get a list of all filepaths for this variable
        var_paths = list(data_dir.glob(f'*{var}.txt'))
        # Sort filepaths by timestamp
        var_paths.sort(key=file_path_to_int)
        var_df = None


        for file_path in var_paths:
            file_str = str(file_path)
            file_idx = int(file_path.parts[-1].split("_")[0])
            points = np.loadtxt(file_str)

            if var in ['x','v']:
                shape = (points.shape[0] // 2, 2)
                #shape = (len(points) // 2, 2)

            elif var in ['F','C']:
                #shape = (points.shape[0] // 2, 2, 2)
                # We want a 2-D shape in the output
                shape = (points.shape[0] // 2, 4)

            elif var in ['timestep','Jp', 'lame']:
                shape = points.shape

            elif var == "mass":
                shape = (resolution, resolution)

            else: # var == velocity
                #shape = (res, res, 2)
                # We want a 2-D shape in the output
                shape = (resolution * resolution, 2)

            points = points.reshape(shape)
            index = np.ones(points.shape[0]) * file_idx
            # Index for data validation
            points = np.c_[points, index] # concatenate idx

            
            points = pd.DataFrame(points)

            if var_df is not None:
                var_df = pd.concat([var_df, points])
            else:
                var_df = points

        dat[var] = var_df
    print(" Done")
    with open(sim_dict_path, 'wb') as writer:
        pkl.dump(dat, writer, protocol=pkl.HIGHEST_PROTOCOL)
    print(f"Saved Results to {sim_dict_path}")
    return dat

def concat_dataframe(verbose=False):
    '''       
        x:         (n, 2)   Position
        v:         (n, 2)   Velocity
        F:         (n, 2)   Deformation Gradient
        C:         (n, 2)   Affine momentum from APIC
        Jp:        (n,)     Determinant of the deformation gradient (i.e. volume)
        lame:      (n, 2)   *Parameters of interest
        velocity:  (res*res, 2)   
        mass:      (resolution, 2)
        timestep:  (n,)
    '''
    data_dict = parse_text_files()
    print("Converting Dict to DataFrame")
    paths = [Path('results.csv'),Path("velocity.csv"), Path("mass.csv")]
    res_csv, vel_csv, mass_csv = paths
    if res_csv.exists():
        for i, path in enumerate(paths):
            paths[i] = pd.read_csv(path)
        return paths
        
    params = 'x,v,F,C,Jp,lame,velocity,mass,timestep'
    params = params.split(',')
    if verbose:
        for key, value in data_dict.items():
            #print(f'\n{key}', value.describe(),sep='\n')
            print(key, value.shape,sep='\t')
            pass
    
    # Rename columns
    for var in params:
        last_col_num = len(data_dict[var].columns) - 1
        #data_dict[var].rename({last_col_num:"index"}, axis=1, inplace=True)
        data_dict[var].drop(columns=[last_col_num], inplace=True)
        if data_dict[var].shape[1] > 1:
            col_preffix = (var+'_{}').format
        else:
            col_preffix = (var).format
        data_dict[var].rename(col_preffix, axis=1, inplace=True)


    
    sub_df = pd.concat([data_dict['x'],data_dict['v']], axis=1)
    # concat all df except 'x' and 'v'
    for var in params:
        # Cannot concat with velocity and mass since nrow differ
        if var not in ['x','v','velocity','mass']:
            sub_df = pd.concat([sub_df,data_dict[var]], axis=1)
    
    print("Saving CSV files")
    sub_df.to_csv(res_csv)
    data_dict['mass'].to_csv(mass_csv)
    data_dict['velocity'].to_csv(vel_csv)

    return sub_df, data_dict['velocity'], data_dict['mass']

if __name__ == "__main__":
    concat_dataframe()

            





