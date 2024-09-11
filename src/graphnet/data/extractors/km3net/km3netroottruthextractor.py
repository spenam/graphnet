"""Code to extract the truth event information from the KM3NeT ROOT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import km3io as ki

#from .weight_events_oscprob import compute_evt_weight


from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    classifier_column_creator,
    create_unique_id,
    create_unique_id_filetype,
    xyz_dir_to_zen_az,
    assert_no_uint_values,
    filter_None_NaN,
)


class KM3NeTROOTTruthExtractor(KM3NeTROOTExtractor):
    """Class for extracting the truth information from a file."""

    def __init__(self, name: str = "truth"):
        """Initialize the class to extract the truth information."""
        super().__init__(name)

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract truth event information as a dataframe."""
        truth_df = self._extract_truth_dataframe(file)
        truth_df = assert_no_uint_values(truth_df)  # asserts the data format

        return truth_df

    def _extract_truth_dataframe(self, file: Any) -> Any:
        """Extract truth information from a file and returns a dataframe.

        Args:
            file (Any): The file from which to extract truth information.

        Returns:
            pd.DataFrame: A dataframe containing truth information.
        """
        nus_flavor = [12, 14, 16]
        padding_value = 99999999.0
        if len(file.mc_trks.E[0]>0):
            primaries = file.mc_trks[:, 0]

            if abs(np.array(primaries.pdgid)[0]) not in nus_flavor:
                # it is a muon file
                # in muon files the first entry is a 81 particle, with no physical meaning
                primaries = file.mc_trks[:, 0]
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])#primaries_jshower.E
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])#primaries_jshower.pos_x
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])#primaries_jshower.pos_y
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])#primaries_jshower.pos_z
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])#primaries_jshower.dir_x
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])#primaries_jshower.dir_y
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])#primaries_jshower.dir_z
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                )
                    
                #check if if has a jmuon reconstruction
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])#primaries_jmuon.E
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])#primaries_jmuon.pos_x
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])#primaries_jmuon.pos_y
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])#primaries_jmuon.pos_z
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])#primaries_jmuon.dir_x
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])#primaries_jmuon.dir_y
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])#primaries_jmuon.dir_z
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                )



                # construct some quantities
                zen_truth, az_truth = xyz_dir_to_zen_az(
                    np.array(primaries.dir_x),
                    np.array(primaries.dir_y),
                    np.array(primaries.dir_z),
                )
                part_dir_x, part_dir_y, part_dir_z = (
                    np.array(primaries.dir_x),
                    np.array(primaries.dir_y),
                    np.array(primaries.dir_z),
                )
                #unique_id = create_unique_id(
                #    np.array(file.run_id),
                #    np.array(file.frame_index),
                #    np.array(file.trigger_counter),
                #)
                unique_id = create_unique_id_filetype(
                    np.array(primaries.pdgid),
                    np.array(primaries.E),
                    np.array(padding_value * np.ones(len(primaries.pos_x))),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
                )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )
                livetime= float(file.header.livetime.numberOfSeconds)
                daq = float(file.header.DAQ.livetime)

                dict_truth = {
                    "pdgid": np.array(primaries.pdgid),
                    "vrx_x": np.array(primaries.pos_x),
                    "vrx_y": np.array(primaries.pos_y),
                    "vrx_z": np.array(primaries.pos_z),
                    "zenith": zen_truth,
                    "azimuth": az_truth,
                    "part_dir_x": part_dir_x,
                    "part_dir_y": part_dir_y,
                    "part_dir_z": part_dir_z,
                    "Energy": np.array(primaries.E),
                    "Bj_x": padding_value * np.ones(len(primaries.pos_x)),
                    "Bj_y": padding_value * np.ones(len(primaries.pos_x)),
                    "i_chan": padding_value * np.ones(len(primaries.pos_x)),
                    "is_cc_flag": padding_value * np.ones(len(primaries.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "w2_gseagen_ps": padding_value * np.ones(len(primaries.pos_x)),
                    "livetime": livetime * np.ones(len(primaries.pos_x)),
                    "DAQ": daq * np.ones(len(primaries.pos_x)),
                    "n_gen": padding_value * np.ones(len(primaries.pos_x)),
                    "w2": padding_value * np.ones(len(primaries.pos_x)),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "w_osc": (daq/livetime) * np.ones(len(primaries.pos_x)),
                }

            else:
                # the particle is a neutrino
                zen_truth, az_truth = xyz_dir_to_zen_az(
                    np.array(primaries.dir_x),
                    np.array(primaries.dir_y),
                    np.array(primaries.dir_z),
                )
                part_dir_x, part_dir_y, part_dir_z = (
                    np.array(primaries.dir_x),
                    np.array(primaries.dir_y),
                    np.array(primaries.dir_z),
                )

                livetime = float(file.header.DAQ.livetime)
                n_gen = 1/file.w[:,3]
                
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                
                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])#primaries_jshower.E
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])#primaries_jshower.pos_x
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])#primaries_jshower.pos_y
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])#primaries_jshower.pos_z
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])#primaries_jshower.dir_x
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])#primaries_jshower.dir_y
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])#primaries_jshower.dir_z
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                )
                
                
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])#primaries_jmuon.E
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])#primaries_jmuon.pos_x
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])#primaries_jmuon.pos_y
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])#primaries_jmuon.pos_z
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])#primaries_jmuon.dir_x
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])#primaries_jmuon.dir_y
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])#primaries_jmuon.dir_z
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                )

                #unique_id = create_unique_id(
                #    np.array(file.run_id),
                #    np.array(file.frame_index),
                #    np.array(file.trigger_counter),
                #)
                unique_id = create_unique_id_filetype(
                    np.array(primaries.pdgid),
                    np.array(primaries.E),
                    np.array(np.array(file.w2list[:, 10] == 2)),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
                )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )

                dict_truth = {
                    "pdgid": np.array(primaries.pdgid),
                    "vrx_x": np.array(primaries.pos_x),
                    "vrx_y": np.array(primaries.pos_y),
                    "vrx_z": np.array(primaries.pos_z),
                    "zenith": zen_truth,
                    "azimuth": az_truth,
                    "part_dir_x": part_dir_x,
                    "part_dir_y": part_dir_y,
                    "part_dir_z": part_dir_z,
                    "Energy": np.array(primaries.E),
                    "Bj_x": np.array(file.w2list[:, 7]),
                    "Bj_y": np.array(file.w2list[:, 8]),
                    "i_chan": np.array(file.w2list[:, 9]),
                    "is_cc_flag": np.array(file.w2list[:, 10] == 2),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "w2_gseagen_ps": np.array(file.w2list[:, 0]),
                    "w2": np.array(file.w[:, 1]),
                    "livetime": livetime * np.ones(len(primaries.pos_x)),
                    "DAQ": livetime * np.ones(len(primaries.pos_x)),
                    "n_gen": n_gen,
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    #"w_osc": compute_evt_weight(np.array(primaries.pdgid),np.array(primaries.E),np.array(primaries.dir_z),np.array(file.w2list[:, 10] == 2),np.array(file.w[:, 1]),n_gen,livetime*np.ones(len(primaries.pos_x))),
                    "w_osc": padding_value * np.ones(len(primaries.pos_x)),
                }

            truth_df = pd.DataFrame(dict_truth)
            is_muon, is_track, is_noise, is_data = classifier_column_creator(
                np.array(dict_truth["pdgid"]), np.array(dict_truth["is_cc_flag"])
            )
            truth_df["is_muon"] = is_muon
            truth_df["is_track"] = is_track
            truth_df["is_noise"] = is_noise
            truth_df["is_data"] = is_data
        else:

            if file.header['calibration']=="dynamical": #data file
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])#primaries_jshower.E
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])#primaries_jshower.pos_x
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])#primaries_jshower.pos_y
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])#primaries_jshower.pos_z
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])#primaries_jshower.dir_x
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])#primaries_jshower.dir_y
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])#primaries_jshower.dir_z
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                )
                    
                #check if if has a jmuon reconstruction
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])#primaries_jmuon.E
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])#primaries_jmuon.pos_x
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])#primaries_jmuon.pos_y
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])#primaries_jmuon.pos_z
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])#primaries_jmuon.dir_x
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])#primaries_jmuon.dir_y
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])#primaries_jmuon.dir_z
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                )



                # construct some quantities
                zen_truth, az_truth = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E))
                part_dir_x, part_dir_y, part_dir_z = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E)),padding_value * np.ones(len(primaries_jmuon.E))
                #unique_id = create_unique_id(
                #    np.array(file.run_id),
                #    np.array(file.frame_index),
                #    np.array(file.trigger_counter),
                #)
                unique_id = create_unique_id_filetype(
                    99 * np.ones(len(primaries_jmuon.E),dtype=int),
                    np.array(padding_value * np.ones(len(primaries_jmuon.pos_x))),
                    np.array(padding_value * np.ones(len(primaries_jmuon.pos_x))),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
                )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )
                daq = float(file.header.DAQ.livetime)
                livetime= float(file.header.DAQ.livetime)

                dict_truth = {
                    "pdgid": 99 * np.ones(len(primaries_jmuon.E),dtype=int),
                    "vrx_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "zenith": padding_value * np.ones(len(primaries_jmuon.E)),
                    "azimuth": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Bj_x": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "Bj_y": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "i_chan": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "is_cc_flag": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "w2_gseagen_ps": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "livetime": livetime * np.ones(len(primaries_jmuon.pos_x)),
                    "DAQ": daq * np.ones(len(primaries_jmuon.pos_x)),
                    "n_gen": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "w2": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "w_osc": np.ones(len(primaries_jmuon.pos_x)),
                }

            else:
                # the event is pure noise
                primaries_jshower = ki.tools.best_jshower(file.trks)
                primaries_jmuon = ki.tools.best_jmuon(file.trks)

                #check if if has a jshower reconstruction
                primaries_jshower_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.E])#primaries_jshower.E
                primaries_jshower_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_x])#primaries_jshower.pos_x
                primaries_jshower_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_y])#primaries_jshower.pos_y
                primaries_jshower_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.pos_z])#primaries_jshower.pos_z
                primaries_jshower_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_x])#primaries_jshower.dir_x
                primaries_jshower_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_y])#primaries_jshower.dir_y
                primaries_jshower_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jshower.dir_z])#primaries_jshower.dir_z
                zen_jshower, az_jshower = xyz_dir_to_zen_az(
                    primaries_jshower_dir_x,
                    primaries_jshower_dir_y,
                    primaries_jshower_dir_z,
                )
                    
                #check if if has a jmuon reconstruction
                primaries_jmuon_E = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.E])#primaries_jmuon.E
                primaries_jmuon_pos_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_x])#primaries_jmuon.pos_x
                primaries_jmuon_pos_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_y])#primaries_jmuon.pos_y
                primaries_jmuon_pos_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.pos_z])#primaries_jmuon.pos_z
                primaries_jmuon_dir_x = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_x])#primaries_jmuon.dir_x
                primaries_jmuon_dir_y = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_y])#primaries_jmuon.dir_y
                primaries_jmuon_dir_z = np.array([filter_None_NaN(element, padding_value) for element in primaries_jmuon.dir_z])#primaries_jmuon.dir_z
                zen_jmuon, az_jmuon = xyz_dir_to_zen_az(
                    primaries_jmuon_dir_x,
                    primaries_jmuon_dir_y,
                    primaries_jmuon_dir_z,
                )



                # construct some quantities
                zen_truth, az_truth = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E))
                part_dir_x, part_dir_y, part_dir_z = padding_value * np.ones(len(primaries_jmuon.E)), padding_value * np.ones(len(primaries_jmuon.E)),padding_value * np.ones(len(primaries_jmuon.E))
                #unique_id = create_unique_id(
                #    np.array(file.run_id),
                #    np.array(file.frame_index),
                #    np.array(file.trigger_counter),
                #)
                unique_id = create_unique_id_filetype(
                    np.zeros(len(primaries_jmuon.E),dtype=int),
                    np.array(padding_value * np.ones(len(primaries_jmuon.pos_x))),
                    np.array(padding_value * np.ones(len(primaries_jmuon.pos_x))),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.id),
                )
                evt_id, run_id, frame_index, trigger_counter = (
                    np.array(file.id),
                    np.array(file.run_id),
                    np.array(file.frame_index),
                    np.array(file.trigger_counter),
                )

                daq = float(file.header.DAQ.livetime)
                livetime= float(file.header.K40) # simulated noise

                dict_truth = {
                    "pdgid": np.zeros(len(primaries_jmuon.E),dtype=int),
                    "vrx_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "vrx_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "zenith": padding_value * np.ones(len(primaries_jmuon.E)),
                    "azimuth": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_x": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_y": padding_value * np.ones(len(primaries_jmuon.E)),
                    "part_dir_z": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Energy": padding_value * np.ones(len(primaries_jmuon.E)),
                    "Bj_x": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "Bj_y": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "i_chan": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "is_cc_flag": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "jshower_E": primaries_jshower_E,
                    "jshower_pos_x": primaries_jshower_pos_x,
                    "jshower_pos_y": primaries_jshower_pos_y,
                    "jshower_pos_z": primaries_jshower_pos_z,
                    "jshower_dir_x": primaries_jshower_dir_x,
                    "jshower_dir_y": primaries_jshower_dir_y,
                    "jshower_dir_z": primaries_jshower_dir_z,
                    "jshower_zenith": zen_jshower,
                    "jshower_azimuth": az_jshower,
                    "jmuon_E": primaries_jmuon_E,
                    "jmuon_pos_x": primaries_jmuon_pos_x,
                    "jmuon_pos_y": primaries_jmuon_pos_y,
                    "jmuon_pos_z": primaries_jmuon_pos_z,
                    "jmuon_dir_x": primaries_jmuon_dir_x,
                    "jmuon_dir_y": primaries_jmuon_dir_y,
                    "jmuon_dir_z": primaries_jmuon_dir_z,
                    "jmuon_zenith": zen_jmuon,
                    "jmuon_azimuth": az_jmuon,
                    "n_hits": np.array(file.n_hits),
                    "w2_gseagen_ps": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "livetime": livetime * np.ones(len(primaries_jmuon.pos_x)),
                    "DAQ": daq * np.ones(len(primaries_jmuon.pos_x)),
                    "n_gen": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "w2": padding_value * np.ones(len(primaries_jmuon.pos_x)),
                    "run_id": run_id,
                    "evt_id": evt_id,
                    "frame_index": frame_index,
                    "trigger_counter": trigger_counter,
                    "event_no": np.array(unique_id).astype(int),
                    "w_osc": (daq/livetime) * np.ones(len(primaries_jmuon.pos_x)),
                }

                

            truth_df = pd.DataFrame(dict_truth)
            is_muon, is_track, is_noise, is_data = classifier_column_creator(
                np.array(dict_truth["pdgid"]), np.array(dict_truth["is_cc_flag"])
            )
            truth_df["is_muon"] = is_muon
            truth_df["is_track"] = is_track
            truth_df["is_noise"] = is_noise
            truth_df["is_data"] = is_data
            

        return truth_df
