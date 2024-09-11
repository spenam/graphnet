"""Code with some functionalities for the extraction."""
from typing import List, Tuple, Any, Union, Literal

import numpy as np
import pandas as pd
import math

def create_unique_id_filetype(
    pdg_id: List[int],
    energy: List[float],
    is_cc_flag: List[int],
    run_id: List[int],
    frame_index: List[int],
    evt_id: List[int],
) -> List[str]:
    """Creating a code for each type of flavor and energy range.""" #TODO check if the enery ranges are suitable for ARCA
    code_dict = {   
                    'atm_muon': 1,
                    'noise': 2,
                    'data': 3,
                    'elec_1_100': 4,
                    'elec_100_500': 5,
                    'elec_500_10000': 6,
                    'muon_1_100': 7,
                    'muon_100_500': 8,
                    'muon_500_10000': 9,
                    'tau_1_100': 10,
                    'tau_100_500': 11,
                    'tau_500_10000': 12,
                    'anti_elec_1_100': 13,
                    'anti_elec_100_500': 14,
                    'anti_elec_500_10000': 15,
                    'anti_muon_1_100': 16,
                    'anti_muon_100_500': 17,
                    'anti_muon_500_10000': 18,
                    'anti_tau_1_100': 19,
                    'anti_tau_100_500': 20,
                    'anti_tau_500_10000': 21,
                    'NC_1_100': 22,
                    'NC_100_500': 23,
                    'NC_500_10000': 24,
                    'anti_NC_1_100': 25,
                    'anti_NC_100_500': 26,
                    'anti_NC_500_10000': 27,
                    }
    
    unique_id = []
    for i in range(len(pdg_id)):
        #compute the file_id
        #for electrons
        if pdg_id[i] == 12:
            if energy[i] < 100:
                file_id = code_dict['elec_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['elec_100_500']
            else:
                file_id = code_dict['elec_500_10000']
        #for muons
        if pdg_id[i] == 14:
            if energy[i] < 100:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['muon_1_100']
                else:
                    file_id = code_dict['NC_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                if is_cc_flag[i] == 1:
                    file_id = code_dict['muon_100_500']
                else:
                    file_id = code_dict['NC_100_500']
            else:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['muon_500_10000']
                else:
                    file_id = code_dict['NC_500_10000']
        #for taus
        if pdg_id[i] == 16:
            if energy[i] < 100:
                file_id = code_dict['tau_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['tau_100_500']
            else:
                file_id = code_dict['tau_500_10000']
        #for anti-electrons
        if pdg_id[i] == -12:
            if energy[i] < 100:
                file_id = code_dict['anti_elec_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['anti_elec_100_500']
            else:
                file_id = code_dict['anti_elec_500_10000']
        #for anti-muons
        if pdg_id[i] == -14:
            if energy[i] < 100:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['anti_muon_1_100']
                else:
                    file_id = code_dict['anti_NC_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                if is_cc_flag[i] == 1:
                    file_id = code_dict['anti_muon_100_500']
                else:
                    file_id = code_dict['anti_NC_100_500']
            else:
                if is_cc_flag[i] == 1:
                    file_id = code_dict['anti_muon_500_10000']
                else:
                    file_id = code_dict['anti_NC_500_10000']
        #for anti-taus
        if pdg_id[i] == -16:
            if energy[i] < 100:
                file_id = code_dict['anti_tau_1_100']
            elif (energy[i] >= 100) & (energy[i] < 500):
                file_id = code_dict['anti_tau_100_500']
            else:
                file_id = code_dict['anti_tau_500_10000']
        #for atmospheric muons
        if pdg_id[i] not in [12, 14, 16, -12, -14, -16, 0, 99]:
            file_id = code_dict['atm_muon']
        #for noise
        if pdg_id[i] == 0:
            file_id = code_dict['noise']
        #for data
        if pdg_id[i] == 99:
            file_id = code_dict['data']

        #compute the unique_id as file_id + run_id + evt_id 
        unique_id.append(str(file_id) + '0' + str(run_id[i]) + '0' + str(evt_id[i]))

    return unique_id

def create_unique_id(
    run_id: List[int],
    frame_index: List[int],
    trigger_counter: List[int],
) -> List[str]:
    """Create unique ID as run_id*1e9 + frame_index*1e6 + trigger_counter,
    hopefully this won't create clashes of events having a the same unique ID
    which is veeeery unlikely but could happen"""
    unique_id = []
    for i in range(len(run_id)):
        unique_id.append(
            run_id[i]*1e9 + 
            frame_index[i]*1e6 +
            trigger_counter[i]
        )

    return unique_id

def filter_None_NaN(
    value: Union[float, None, Literal[math.nan]],
    padding_value: float,
) -> float:
    """Removes None or Nan values and transforms it to padding float value."""
    if value is None or math.isnan(value):
        return padding_value
    else:
        return value

def create_unique_id_dbang(
    energy: List[float],
    pos_x: List[float],
    ids: List[int],
) -> List[str]:
    """Create unique ID for double bang events."""
    unique_id = []
    for i in range(len(energy)):
        unique_id.append(
            str(ids[i])
            + str(int(1000*energy[i]))
            + str(int(abs(1000*pos_x[i])))
        )
    return unique_id


def xyz_dir_to_zen_az(
    dir_x: List[float],
    dir_y: List[float],
    dir_z: List[float],
) -> Tuple[List[float], List[float]]:
    """Convert direction vector to zenith and azimuth angles."""
    # Compute zenith angle (elevation angle)
    zenith = np.arccos(dir_z)  # zenith angle in radians

    # Compute azimuth angle
    azimuth = np.arctan2(dir_y, dir_x)  # azimuth angle in radians
    az_centered = azimuth + np.pi * np.ones(
        len(azimuth)
    )  # Center the azimuth angle around zero

    return zenith, az_centered


def classifier_column_creator(
    pdgid: np.ndarray,
    is_cc_flag: List[int],
) -> Tuple[List[int], List[int]]:
    """Create helpful columns for the classifier."""
    is_muon = np.zeros(len(pdgid), dtype=int)
    is_track = np.zeros(len(pdgid), dtype=int)
    is_noise = np.zeros(len(pdgid), dtype=int)
    is_data = np.zeros(len(pdgid), dtype=int)
    
    #TODO add tau topology
    """
    primaries = f.mc_trks[:,0]
    secondaries = f.mc_trks
    
    print("%"*20)
    tau=abs(primaries.pdgid)==16
    tau_track=np.any(np.abs(secondaries.pdgid)==13,axis=1)
    result=np.logical_and(tau,tau_track)
    print(secondaries.pdgid[result])
    for i in secondaries.pdgid[result]: # this will get all track like taus, we could add a new column as 'tau_topology'
        print(i)

    """

    is_muon[abs(pdgid) == 13] = 1
    is_muon[pdgid == 81] = 1
    is_track[abs(pdgid) == 13] = 1
    is_track[pdgid == 81] = 1
    is_track[(abs(pdgid) == 14) & (is_cc_flag == 1)] = 1
    is_noise[pdgid == 0] = 1
    is_data[pdgid == 99] = 1

    return is_muon, is_track, is_noise, is_data


def creating_time_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the event time so that the first hit has zero in time."""
    #df = df.sort_values(by=["event_no", "t"])
    df["min_t"] = df.groupby("event_no")["t"].transform("min")
    df["t"] = df["t"] - df["min_t"]
    df = df.drop(["min_t"], axis=1)

    return df


def assert_no_uint_values(df: pd.DataFrame) -> pd.DataFrame:
    """Assert no format no supported by sqlite is in the data."""
    for column in df.columns:
        if df[column].dtype == "uint32":
            df[column] = df[column].astype("int32")
        elif df[column].dtype == "uint64":
            df[column] = df[column].astype("int64")
    return df

def mask_saturated_pmts(df: pd.DataFrame, tot:int = 254) -> pd.DataFrame:
    """Mask saturated PMTs in each event independently"""
    condition_rows = df[df['tot'] > tot][['event_no', 'channel_id', 'dom_id']]
    merged_df = df.merge(condition_rows, on=['event_no', 'channel_id', 'dom_id'], how='left', indicator=True)
    filtered_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    return filtered_df


