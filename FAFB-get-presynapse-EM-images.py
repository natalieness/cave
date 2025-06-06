#%%
import cloudvolume 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
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
