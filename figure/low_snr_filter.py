import numpy as np
import os
import copy as cp
import shutil

snr_thre = 0.1
file_dir = r""


modified_dir = os.path.join(file_dir, "snr filtered")
os.makedirs(modified_dir, exist_ok=True)

with open(os.path.join(file_dir, 'F.npy'), 'rb') as f:
    Fcell: np.ndarray = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)
        
with open(os.path.join(file_dir, 'Fneu.npy'), 'rb') as f:
    Fneu: np.ndarray = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)
    
with open(os.path.join(file_dir, 'spks.npy'), 'rb') as f:
    DeconvSignal: np.ndarray = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)
    
with open(os.path.join(file_dir, 'iscell.npy'), 'rb') as f:
    iscell: np.ndarray = np.load(f, allow_pickle=True)

with open (os.path.join(file_dir, "stat.npy"), "rb") as f:
    stat = np.load(f, allow_pickle=True)
    
dF = Fcell.copy() - 0.7 * Fneu
snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)  

filtered_snr = np.where(snr >= snr_thre)[0]
print(f"Downsampled from {len(snr)} to {len(filtered_snr)} neurons after low SNR filtering.")

Fcell_filtered = Fcell[filtered_snr, :]
Fneu_filtered = Fneu[filtered_snr, :]
dconvolv_filtered = DeconvSignal[filtered_snr, :]
is_cell_filtered = iscell[filtered_snr, :]
stat_modi = []
for i in filtered_snr:
    stat_modi.append(stat[i])

with open(os.path.join(modified_dir, 'F.npy'), 'wb') as f:
    np.save(f, Fcell_filtered)
    
with open(os.path.join(modified_dir, 'Fneu.npy'), 'wb') as f:
    np.save(f, Fneu_filtered)
    
with open(os.path.join(modified_dir, 'spks.npy'), 'wb') as f:
    np.save(f, dconvolv_filtered)
    
with open(os.path.join(modified_dir, 'iscell.npy'), 'wb') as f:
    np.save(f, is_cell_filtered)

shutil.copy(os.path.join(file_dir, 'ops.npy'), os.path.join(modified_dir, 'ops.npy'))

with open(os.path.join(modified_dir, 'stat.npy'), 'wb') as f:
    np.save(f, stat_modi)
