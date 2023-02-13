from dataclasses import dataclass, field
from typing import Union, Dict, Any, ClassVar, Optional, Tuple, List
from datetime import datetime
import numpy as np

@dataclass
class PartSignal:  # todo abc
    """
    Class object creating a dictionnary including different signals linked to participant id.
    """

    signal_dict: Dict[str, Union[np.ndarray, float]]
    partID: int
    date_measure: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        self._original_signal_names = self.metadata['SignalName']

    @property
    def __getitem__(self, item):
        return self.signal_dict[item]

    def get_signal_dict(self):
        return self.signal_dict

    def get_signal(self, signal_name):
        signal_id = np.where(signal_dict['SignalName']==signal_name)[0][0]
        return self.signal_dict['Signal'][signal_id,:]

    def get_all_signal_names(self):
        return list(self.signal_dict.keys())

    def get_all_signal_original_names(self):
        return list(self._original_signal_names)

    def get_all_signal(self):
        nb_signal = len(list(self.signal_dict.keys()))
        signal_vars = []
        for signal_vars in self.signal_dict.values():
            signal_vars.append(self.get_signal(signal_vars))
        return signal_vars


