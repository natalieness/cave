from caveclient import CAVEclient
import cloudvolume
from matplotlib import pyplot as plt
import numpy as np
import meshparty 


client = CAVEclient(
    datastack_name="prieto_godino_fly_larva",
    server_address="https://proofreading.zetta.ai"
)

client.info.get_datastacks()

#to view information about the datastack
client.info.get_datastack_info()

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

#%% Skeleton access 
skeleton_path = 'precomputed://middleauth+https://data.proofreading.zetta.ai/skeletoncache/api/v1/prieto_godino_fly_larva/precomputed/skeleton'

cv_skeleton = cloudvolume.CloudVolume(skeleton_path, use_https=True, fill_missing=True)
# %%
# Initialise the volume
vol = cloudvolume.CloudVolume("graphene://middleauth+https://data.proofreading.zetta.ai/segmentation/table/pg_fly_larva_aff_0_38_whole_v0", use_https=True)

# %% get root ids, supervoxels and nodes from chunkedgraph
# get root id for a supervoxel
sv_id = 648518346369003406
root_id = client.chunkedgraph.get_root_id(supervoxel_id=sv_id)

#get supervoxels for root id
supervoxels = client.chunkedgraph.get_leaves(root_id=root_id)

# see https://caveconnectome.github.io/CAVEclient/api/chunkedgraph/#caveclient.chunkedgraph.ChunkedGraphClient

# %%
