#%%
import cloudvolume 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pymaid
#%%
em_path = "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig"

cv_em = cloudvolume.CloudVolume(em_path, use_https=True, fill_missing=True, mip=0)
cv_em.shape
vol= cv_em[65000:66000, 78400:79400, 2540:2550]
print(vol.shape)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vol[..., 0, 0].T, cmap="gray") 
#transpose to match the neuroglancer view with same coordinates

#find presynapses in the volume 
# %%
# Connect to the VFB CATMAID server hosting the FAFB data
rm = pymaid.connect_catmaid(server="https://fafb.catmaid.virtualflybrain.org/", api_token=None, max_threads=10)

#%%
random_skid = 22976
neuron = pymaid.get_neurons(random_skid)
#presyn_NT = [neu.connectors[neu.connectors.type==0] for neu in neuron]
presyn = neuron.connectors[neuron.connectors.type==0]

presyn1 = [presyn.x[0]  , presyn.y[0], presyn.z[0]]
# %%

vol= cv_em[presyn1[0]:presyn1[0]+1000, presyn1[1]:presyn1[1]+1000, presyn1[2]:presyn1[2]+100]
print(vol.shape)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vol[..., 0, 0].T, cmap="gray") 

# %%
