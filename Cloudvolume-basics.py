"""

Note, the environment to use is intheclouds

"""
#%%
import cloudvolume
from matplotlib import pyplot as plt
import numpy as np
import meshparty 

#%%
# The cloud path for the 630 release version (1st public release)
seg_path = "precomputed://gs://flywire_v141_m630"

# The cloud path for the 783 release version (2nd public release)
# seg_path = "precomputed://gs://flywire_v141_m783"
#%%

### Segmentation access
#instantiate cloudvolume instance
cv_seg = cloudvolume.CloudVolume(seg_path, use_https=True, fill_missing=True)

print(cv_seg.resolution)
print(cv_seg.bounds)
#  data can be sliced like in numpy 
vol = cv_seg[32500:33000, 12500:13000, 2500:2510]
vol.shape

# Convenience function to map neuron IDs to nice colors
def remap_seg(seg, b=8, seed=23):    
    u_ids = np.unique(seg)
    np.random.seed(seed)
    np.random.shuffle(u_ids)
    mapping = np.vectorize(dict(zip(u_ids, np.random.randint(0, 2**b-1, size=len(u_ids)))).get)
    
    remapped_seg = mapping(seg).astype(int)
    
    return remapped_seg

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(remap_seg(vol[..., 0, 0]), cmap="nipy_spectral")

# %% EM image access

# The cloud path the EM remains the same between versions
em_path = "precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"
cv_em = cloudvolume.CloudVolume(em_path, use_https=True, fill_missing=True, mip=1)
vol = cv_em[65000:66000, 25000:26000, 2500:2510]
print(vol.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vol[..., 0, 0], cmap="gray")



# %% Mesh access (available from segmentation instance)

mesh = cv_seg.mesh.get(720575940631693610)[720575940631693610]

mesh_lod2 = cv_seg.mesh.get(720575940631693610, lod=2)[720575940631693610]
mesh.vertices.shape, mesh_lod2.vertices.shape

# %% with mesh party

from meshparty import trimesh_io
mesh_meta = trimesh_io.MeshMeta(cv_path=seg_path, cache_size=10)

%time mesh = mesh_meta.mesh(seg_id=720575940631693610)
%time mesh = mesh_meta.mesh(seg_id=720575940631693610)