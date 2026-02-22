import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import torch

from joblib import load 
from classicalgsg.ClassicalGSG.src.classicalgsg.molreps_models.gsg import GSG
from classicalgsg.ClassicalGSG.src.classicalgsg.classicalgsg import CGenFFGSG, OBFFGSG
from classicalgsg.ClassicalGSG.src.classicalgsg.molreps_models.utils import scop_to_boolean

PRETRAINED_MODEL_PATH = './classicalgsg/ClassicalGSG/src/classicalgsg/pretrained_models/'
CGENFFGSG_MODEL = 'model_4_zfs_CGenFF.pkl'
CGENFFGSG_SCALAR = 'std_scaler_CGenFF.sav'
OBFFGSG_MODEL = 'model_4_zfs_MMFF.pkl'
OBFFGSG_SCALAR = 'std_scaler_MMFF.sav'
FORCEFIELD = 'MMFF94'

class ClassicalGSGHandler:
    """Handles interaction with Both types of LogP predictions"""

    AVAILABLE_PROPERTIES = ["LogP_MMFF","LogP_CGenFF"] #A little different since both models predict LogP

    def __init__(self):
        self.device = torch.device('cpu')
        self.max_wavelet_scale = 4
        self.scattering_operators = '(z,f,s)'
        self.gsg = GSG(self.max_wavelet_scale, scop_to_boolean(self.scattering_operators))

        self.models = {"LogP_MMFF":None, "LogP_CGenFF":None}
        self.scalars = {"LogP_MMFF":None, "LogP_CGenFF":None}
        self.mod_classes = {"LogP_MMFF":None, "LogP_CGenFF":None} 

        self._load_models()

    def _load_models(self) -> None:
        self.mod_classes["LogP_MMFF"] = OBFFGSG(self.gsg, structure='2D', AC_type='ACall')
        self.mod_classes["LogP_CGenFF"] = CGenFFGSG(self.gsg, structure='2D', AC_type='AC36')

        self.models["LogP_MMFF"] = torch.load(f"{PRETRAINED_MODEL_PATH}/{OBFFGSG_MODEL}")
        self.models["LogP_CGenFF"] = torch.load(f"{PRETRAINED_MODEL_PATH}/{CGENFFGSG_MODEL}")

        self.scalars["LogP_MMFF"] = load(f"{PRETRAINED_MODEL_PATH}/{OBFFGSG_SCALAR}")
        self.scalars["LogP_CGenFF"] = load(f"{PRETRAINED_MODEL_PATH}/{CGENFFGSG_SCALAR}")


    def process_multiple_properties(self, smiles:str, property_list: List[str]) -> Dict[str, Any]:
        if not smiles:
            return {}
        
        # Validate property list
        valid_props = [p for p in property_list if p in self.AVAILABLE_PROPERTIES]

        batch_preds = {}

        try:
            if "LogP_MMFF" in valid_props:
                features = self.mod_classes["LogP_MMFF"].features(smiles, FORCEFIELD)
                if features is not None:
                    reshape = features.reshape((-1, features.shape[0]))

                    transform = self.scalars["LogP_MMFF"].transform(reshape)

                    batch_preds["LogP_MMFF"] = np.squeeze(self.models["LogP_MMFF"].predict(transform.astype(np.float32)))

            if "LogP_CGenFF" in valid_props:
                features = self.mod_classes["LogP_CGenFF"].features(smiles, FORCEFIELD)
                if features is not None:
                    reshape = features.reshape((-1, features.shape[0]))

                    transform = self.scalars["LogP_CGenFF"].transform(reshape)

                    batch_preds["LogP_CGenFF"] = np.squeeze(self.models["LogP_CGenFF"].predict(transform.astype(np.float32)))

            res_entry = {
                "smiles": smiles,
                "status": "success",
                "results": {},
                "error": None
            }

            for prop in valid_props:
                res_entry["results"][prop] = {
                    "property": prop, 
                    "status": "success",
                    "results": float(batch_preds[prop]),
                    "error": None
                }
            
        except Exception as e:
            # Return error objects for the batch
            return {"smiles": smiles, "status": "error", "results": {}, "error": str(e)} 

    def process_multiple_properties_batch(self, smiles_list: List[str], property_list: List[str]) -> List[Dict[str, Any]]:
        res_ls = []
        for smi in smiles_list:
            one_smi_res_obj = self.process_multiple_properties(smi, property_list)
            res_ls.append(one_smi_res_obj)
        return res_ls
        
             

