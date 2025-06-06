from caveclient import CAVEclient
import cloudvolume
from matplotlib import pyplot as plt
import numpy as np
import meshparty 
from meshparty import trimesh_io, trimesh_vtk


client = CAVEclient(
    datastack_name="prieto_godino_fly_larva",
    server_address="https://proofreading.zetta.ai"
)

client.info.get_datastacks()

#to view information about the datastack
client.info.get_datastack_info()

#Note: no skeleton access on this datastack yet!

#%% get existing annotation tables

all_tables = client.annotation.get_tables()
all_tables[0]

table_name = all_tables[0]   # 'ais_analysis_soma'
annotation_id = 100
client.annotation.get_annotation(annotation_ids=annotation_id, table_name=table_name)


# %% EM access 
em_path = 'precomputed://gs://zetta-prieto-godino-fly-larva-001-image/image-v1-iso'
cv_em = cloudvolume.CloudVolume(em_path, use_https=True, fill_missing=True, mip=1)
print(cv_em.resolution)
print(cv_em.bounds)

vol = cv_em[5300:7300,5300:7300,2003]
print(vol.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vol[..., 0, 0], cmap="gray")
# %%
# %% Segmentation access
seg_path = 'graphene://middleauth+https://data.proofreading.zetta.ai/segmentation/table/pg_fly_larva_aff_0_38_whole_v0'
#seg_path = 'graphene://https://data.proofreading.zetta.ai/segmentation/table/pg_fly_larva_aff_0_38_whole_v0'
cv_seg = cloudvolume.CloudVolume(seg_path, use_https=True, fill_missing=True)

vol = cv_seg[5300:7300,5300:7300,2003]

def remap_seg(seg, b=8, seed=23):    
    u_ids = np.unique(seg)
    np.random.seed(seed)
    np.random.shuffle(u_ids)
    mapping = np.vectorize(dict(zip(u_ids, np.random.randint(0, 2**b-1, size=len(u_ids)))).get)
    
    remapped_seg = mapping(seg).astype(int)
    
    return remapped_seg

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(remap_seg(vol[..., 0, 0]), cmap="nipy_spectral")


# %% Mesh access from segmentation instance
# get root id for a supervoxel
sv_id = 648518346369003406
root_id = client.chunkedgraph.get_root_id(supervoxel_id=sv_id)

# get mesh for root id
mesh = cv_seg.mesh.get(root_id)[root_id]
mesh.vertices.shape

#%%
mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                     color=(1, 0, 0),   # RGB (red)
                                     opacity=0.5)       # transparent
trimesh_vtk.render_actors([mesh_actor])


# %% get root ids, supervoxels and nodes from chunkedgraph
# get root id for a supervoxel
sv_id = 648518346369003406
root_id = client.chunkedgraph.get_root_id(supervoxel_id=sv_id)

#get supervoxels for root id
supervoxels = client.chunkedgraph.get_leaves(root_id=root_id)

# see https://caveconnectome.github.io/CAVEclient/api/chunkedgraph/#caveclient.chunkedgraph.ChunkedGraphClient

# %%
