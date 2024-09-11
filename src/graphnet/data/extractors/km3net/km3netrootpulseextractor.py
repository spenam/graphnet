"""Code to extract the pulse map information from the KM3NeT ROOT file."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import awkward as ak
import km3io as ki

from graphnet.data.extractors import Extractor
from .km3netrootextractor import KM3NeTROOTExtractor
from graphnet.data.extractors.km3net.utilities.km3net_utilities import (
    create_unique_id,
    create_unique_id_filetype,
    mask_saturated_pmts,
    assert_no_uint_values,
    creating_time_zero,
)


class KM3NeTROOTPulseExtractor(KM3NeTROOTExtractor):
    """Class for extracting the entire pulse information from a file."""

    def __init__(self, name: str = "pulse_map"):
        """Initialize the class to extract the pulse information."""
        super().__init__(name)

    def __call__(self, file: Any) -> Dict[str, Any]:
        """Extract pulse map information and return a dataframe.

        Args:
            file (Any): The file from which to extract pulse map information.

        Returns:
            Dict[str, Any]: A dictionary containing pulse map information.
        """
        pulsemap_df = self._extract_pulse_map(file)
        pulsemap_df = assert_no_uint_values(pulsemap_df)

        return pulsemap_df

    def _extract_pulse_map(self, file: Any) -> pd.DataFrame:
        """Extract the pulse information and assigns unique IDs.

        Args:
            file (Any): The file from which to extract pulse information.

        Returns:
            pd.DataFrame: A dataframe containing pulse information.
        """
        #unique_id = create_unique_id(
        #    np.array(file.run_id),
        #    np.array(file.frame_index),
        #    np.array(file.trigger_counter),
        #)  # creates the unique_id
        padding_value = 99999999.0
        nus_flavor = [12, 14, 16]
        if len(file.mc_trks.E[0]>0):
            primaries = file.mc_trks[:, 0]
            E = np.array(primaries.E)
            pdgid = np.array(primaries.pdgid)
            if abs(np.array(primaries.pdgid)[0]) not in nus_flavor:
                # it is a muon file
                is_cc_flag= np.array(padding_value * np.ones(len(primaries.pos_x)))
            else:   
                is_cc_flag=np.array(np.array(file.w2list[:, 10] == 2))
        else:
            primaries_jmuon = ki.tools.best_jmuon(file.trks)
            E = padding_value * np.ones(len(primaries_jmuon.E))
            is_cc_flag= np.array(padding_value * np.ones(len(primaries_jmuon.E)))
            if file.header['calibration']=='dynamical': #data file
                pdgid = 99 * np.ones(len(primaries_jmuon.E),dtype=int)
            else:
                pdgid = np.zeros(len(primaries_jmuon.E),dtype=int)
        unique_id = create_unique_id_filetype(
            pdgid,
            E,
            is_cc_flag,
            np.array(file.run_id),
            np.array(file.frame_index),
            np.array(file.id),
        )

        hits = file.hits
        keys_to_extract = [
            "t",
            "pos_x",
            "pos_y",
            "pos_z",
            "dir_x",
            "dir_y",
            "dir_z",
            "tot",
            "trig",
        ]
        mask_saturated_pmts = False
        if mask_saturated_pmts:
            keys_to_extract.append("channel_id")
            keys_to_extract.append("dom_id")

        pandas_df = ak.to_dataframe(hits.arrays(keys_to_extract, library="ak"))
        df = pandas_df.reset_index()
        unique_extended = []
        for index in df["entry"].values:
        #for index in df["index"].values:
            unique_extended.append(int(unique_id[index]))
        df["event_no"] = np.array(unique_extended).astype(int)
        # keep only non saturated pmts or DOMs
        if mask_saturated_pmts:
            df = mask_saturated_pmts(df)
            for k in ["channel_id","dom_id"]:
                if k not in keys_to_extract:
                    df = df.drop(k, axis=1)

        df = df.drop(["entry", "subentry"], axis=1)
        for k in ["tot","trig","channel_id","dom_id","event_no"]:
            if k in df.columns:
                df[k]=df[k].astype('int64')
        df = creating_time_zero(df)
        print(df)

        return df
