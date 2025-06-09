#%%
import cloudvolume 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pymaid
from caveclient import CAVEclient
import fafbseg
import navis


from FAFB_token import FAFB_token

#%%

datastack_name = "flywire_fafb_public"
client = CAVEclient(datastack_name,
                    auth_token = FAFB_token)

#get version
print(f"available version{client.materialize.get_versions()}")

client.materialize.get_tables()
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

# %% find the same neurons in cave -- this doesnt really work? not sure why? 
#can't see the presynaptic site in the EM data? maybe wrong transformation or something to do with the root id stuff?
random_skid = 22976
all_neurons = client.materialize.query_table("proofread_neurons")

random_root = all_neurons[all_neurons["id"] == random_skid]['pt_root_id']
presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": random_root})
postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": random_root})

random_pt_position = presyn_df['pre_pt_position'].loc[11]
# %%
# Connect to the VFB CATMAID server hosting the FAFB data
rm = pymaid.connect_catmaid(server="https://fafb.catmaid.virtualflybrain.org/", api_token=None, max_threads=10)


random_skid = 22976
neuron = pymaid.get_neurons(random_skid)
#presyn_NT = [neu.connectors[neu.connectors.type==0] for neu in neuron]
presyn = neuron.connectors[neuron.connectors.type==0]

number = 76
presyn1 = [presyn.x[number]  , presyn.y[number], presyn.z[number]]


#random_pt_position = all_neurons[all_neurons['id'] == random_skid]['pt_position'].values[0]
random_pt_position = np.array(presyn1) #try from catmaid

#random_pt_position = np.array([random_pt_position[0]/4, random_pt_position[1]/4, random_pt_position[2]/40])
random_pt_position = random_pt_position.reshape(1, 3)
transformed_pos = navis.xform_brain(random_pt_position, source="FAFB14", target="FAFB14raw")

xy_buff = 225
z_buff = 100

def center_position(position, xy_buff, z_buff):
    """Return a bounding box around the position with given buffer."""
    xstart = position[0] - xy_buff // 2
    xend = position[0] + xy_buff // 2
    ystart = position[1] - xy_buff // 2
    yend = position[1] + xy_buff // 2
    zstart = position[2] #- z_buff // 2
    zend = position[2] + z_buff #// 2
    return (xstart, xend, ystart, yend, zstart, zend)

xstart, xend, ystart, yend, zstart, zend = center_position(transformed_pos[0], xy_buff, z_buff)

vol= cv_em[xstart:xend, ystart:yend, zstart:zend]
print(vol.shape)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(vol[..., 0, 0].T, cmap="gray") 
# %%
