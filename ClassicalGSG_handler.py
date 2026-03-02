import os
import sys
import subprocess
import torch
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from joblib import load 

# 2. Add the source directory to path
sys.path.append(os.path.join(os.getcwd(), "ClassicalGSG/src"))

from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import CGenFFGSG, OBFFGSG
from classicalgsg.molreps_models.utils import scop_to_boolean

PRETRAINED_MODEL_PATH = './ClassicalGSG/src/classicalgsg/pretrained_models/'
CGENFFGSG_MODEL = 'model_4_zfs_CGenFF.pkl'
CGENFFGSG_SCALAR = 'std_scaler_CGenFF.sav'
OBFFGSG_MODEL = 'model_4_zfs_MMFF.pkl'
OBFFGSG_SCALAR = 'std_scaler_MMFF.sav'
FORCEFIELD = 'MMFF94'

class ClassicalGSGHandler:
    AVAILABLE_PROPERTIES = ["LogP_MMFF","LogP_CGenFF"]

    def __init__(self):
        self.gsg = GSG(4, scop_to_boolean('(z,f,s)'))
        self.models = {}
        self.scalars = {}
        self.mod_classes = {} 
        self._load_models()

    def _load_models(self) -> None:
        self.mod_classes["LogP_MMFF"] = OBFFGSG(self.gsg, structure='2D', AC_type='ACall')
        self.mod_classes["LogP_CGenFF"] = CGenFFGSG(self.gsg, structure='2D', AC_type='AC36')
        
        # weights_only=False to handle the specific pkl format used here
        self.models["LogP_MMFF"] = torch.load(f"{PRETRAINED_MODEL_PATH}/{OBFFGSG_MODEL}", weights_only=False)
        self.models["LogP_CGenFF"] = torch.load(f"{PRETRAINED_MODEL_PATH}/{CGENFFGSG_MODEL}", weights_only=False)
        
        self.scalars["LogP_MMFF"] = load(f"{PRETRAINED_MODEL_PATH}/{OBFFGSG_SCALAR}")
        self.scalars["LogP_CGenFF"] = load(f"{PRETRAINED_MODEL_PATH}/{CGENFFGSG_SCALAR}")

    def process_multiple_properties(self, smiles:str, property_list: List[str]) -> Dict[str, Any]:
        valid_props = [p for p in property_list if p in self.AVAILABLE_PROPERTIES]
        batch_preds = {}

        for prop in valid_props:
            try:

              
                features = self.mod_classes[prop].features(smiles, FORCEFIELD)
                
                if features is not None:
                  
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    reshape = features.reshape((1, -1))
                    transform = self.scalars[prop].transform(reshape)
                    
                    
                    transform = np.nan_to_num(transform, nan=0.0)

                    pred = self.models[prop].predict(transform.astype(np.float32))
                    batch_preds[prop] = float(np.squeeze(pred))
                else:
                    print(f"Warning: Features for {prop} returned None")

            except Exception as e:
                print(f"DEBUG: Error in {prop} calculation: {e}")
                continue

            res_entry = {
                    "smiles": smiles,
                    "status": "success",
                    "results": {},
                    "error": None
                }
            
            res_entry["results"][prop] = {
                            "property": prop,
                            "status": "success",
                            "results": float(batch_preds[prop]),
                            "error": None
                        }
            
        for tmp_file in ["mol.smi", "mol.mol2"]:
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except:
                            pass # File might be locked, continue anyway
        return res_entry
    
        
        
    def process_multiple_properties_batch(self, smiles_list: List[str], property_list: List[str]) -> List[Dict[str, Any]]:
        res_ls = []
        for smi in smiles_list:
            one_smi_res_obj = self.process_multiple_properties(smi, property_list)
            res_ls.append(one_smi_res_obj)
        return res_ls

if __name__ == '__main__':
    handler = ClassicalGSGHandler()
    # Test with MMFF specifically
    result = handler.process_multiple_properties("CC(=O)Oc1ccccc1C(=O)O", ["LogP_MMFF"]) #For the time being I cannot use LogP_CGenFF as it needs a binary.
    print(result)