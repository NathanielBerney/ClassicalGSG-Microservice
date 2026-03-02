import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from joblib import load 

# --- PATH LOGIC ---
# Get the directory where this handler script lives (classicalgsg/)
BASE_DIR = Path(__file__).resolve().parent

# Add 'ClassicalGSG/src' to sys.path so we can import 'classicalgsg'
SOURCE_DIR = BASE_DIR / "ClassicalGSG" / "src"
sys.path.append(str(SOURCE_DIR))

# Update the Pretrained Model Path
PRETRAINED_MODEL_PATH = SOURCE_DIR / 'classicalgsg' / 'pretrained_models'
# ------------------

# Now that path is set, we can import
from classicalgsg.molreps_models.gsg import GSG  # Fixed 'Clas' to 'classicalgsg'
from classicalgsg.classicalgsg import CGenFFGSG, OBFFGSG
from classicalgsg.molreps_models.utils import scop_to_boolean

CGENFFGSG_MODEL = 'model_4_zfs_CGenFF.pkl'
CGENFFGSG_SCALAR = 'std_scaler_CGenFF.sav'
OBFFGSG_MODEL = 'model_4_zfs_MMFF.pkl'
OBFFGSG_SCALAR = 'std_scaler_MMFF.sav'
FORCEFIELD = 'MMFF94'

class ClassicalGSGHandler:
    AVAILABLE_PROPERTIES = ["LogP_MMFF", "LogP_CGenFF"]

    def __init__(self):
        self.gsg = GSG(4, scop_to_boolean('(z,f,s)'))
        self.models = {}
        self.scalars = {}
        self.mod_classes = {} 
        self._load_models()

    def _load_models(self) -> None:
        self.mod_classes["LogP_MMFF"] = OBFFGSG(self.gsg, structure='2D', AC_type='ACall')
        self.mod_classes["LogP_CGenFF"] = CGenFFGSG(self.gsg, structure='2D', AC_type='AC36')
        
        # Using .joinpath for safer path concatenation
        self.models["LogP_MMFF"] = torch.load(PRETRAINED_MODEL_PATH / OBFFGSG_MODEL, weights_only=False)
        self.models["LogP_CGenFF"] = torch.load(PRETRAINED_MODEL_PATH / CGENFFGSG_MODEL, weights_only=False)
        
        self.scalars["LogP_MMFF"] = load(PRETRAINED_MODEL_PATH / OBFFGSG_SCALAR)
        self.scalars["LogP_CGenFF"] = load(PRETRAINED_MODEL_PATH / CGENFFGSG_SCALAR)

    def process_multiple_properties(self, smiles:str, property_list: List[str]) -> Dict[str, Any]:
        valid_props = [p for p in property_list if p in self.AVAILABLE_PROPERTIES]
        batch_preds = {}
        
        # Initialize the response object outside the loop to avoid local variable errors
        res_entry = {
            "smiles": smiles,
            "status": "success",
            "results": {},
            "error": None
        }

        for prop in valid_props:
            try:
                features = self.mod_classes[prop].features(smiles, FORCEFIELD)
                
                if features is not None:
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    reshape = features.reshape((1, -1))
                    transform = self.scalars[prop].transform(reshape)
                    transform = np.nan_to_num(transform, nan=0.0)

                    pred = self.models[prop].predict(transform.astype(np.float32))
                    val = float(np.squeeze(pred))
                    batch_preds[prop] = val
                    
                    # Store result inside the specific property key
                    res_entry["results"][prop] = {
                        "property": prop,
                        "status": "success",
                        "results": val,
                        "error": None
                    }
                else:
                    print(f"Warning: Features for {prop} returned None")

            except Exception as e:
                print(f"DEBUG: Error in {prop} calculation: {e}")
                res_entry["results"][prop] = {"status": "error", "error": str(e)}
                continue

        # Clean up temporary files created by OpenBabel/RDKit during feature extraction
        for tmp_file in ["mol.smi", "mol.mol2"]:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass 
        return res_entry

    def process_multiple_properties_batch(self, smiles_list: List[str], property_list: List[str]) -> List[Dict[str, Any]]:
        return [self.process_multiple_properties(smi, property_list) for smi in smiles_list]

if __name__ == '__main__':
    handler = ClassicalGSGHandler()
    result = handler.process_multiple_properties("CC(=O)Oc1ccccc1C(=O)O", ["LogP_MMFF"])
    print(result)