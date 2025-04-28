"""

Note, the environment to use is intheclouds

"""
#%%
import cloudvolume
from matplotlib import pyplot as plt
import numpy as np
import meshparty 
from meshparty import trimesh_io, trimesh_vtk

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

#%% try get neuronal bounds 

my_vol = cv_seg[32500:35000, 12500:16000, 2500:2710]
u_ids = np.unique(my_vol)


#%% try get neuronal bounds part 2 (without reloading the data)

print(f"Number of unique IDs in volume: {u_ids.shape[0]}")



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

#%% see whats in the mesh 

verts = mesh_lod2.vertices
verts.shape

#%% plot a mesh 


mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                     color=(1, 0, 0),   # RGB (red)
                                     opacity=0.5)       # transparent
trimesh_vtk.render_actors([mesh_actor])
# %% with mesh party


mesh_meta = trimesh_io.MeshMeta(cv_path=seg_path, cache_size=10)

%time mesh = mesh_meta.mesh(seg_id=720575940631693610)
%time mesh = mesh_meta.mesh(seg_id=720575940631693610)

# %%
local_vertices = mesh.get_local_view(5, pc_align=True, method="kdtree")
# %%
from meshparty import skeleton

# Choose a root point (e.g. the vertex closest to centroid)
root = mesh.vertices[mesh.kdtree.query(mesh.centroid)[1]]

skel = mesh_to_skeleton(mesh, root,
                        remove_duplicate_vertices=True,
                        invalidation_d=5000)
# %%
skel = cv_seg.skeleton.get(720575940631693610)[720575940631693610]
# %%
