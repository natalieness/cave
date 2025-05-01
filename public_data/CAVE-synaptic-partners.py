from caveclient import CAVEclient
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%% initialise client
client = CAVEclient()
client.info.get_datastacks()

#%% initialise example datastack 

datastack_name = "flywire_fafb_public"
client = CAVEclient(datastack_name)

#get version
print(f"available version{client.materialize.get_versions()}")

client.materialize.get_tables()

# %% see proofread neurons
client.materialize.query_table("proofread_neurons")
#centered on backbone - not all neurons in this data have a cell body, e.g. sensory neurons
#nuclei are in nuclei_v1 (from Shang et al)

# %% each table has a description

print(client.materialize.get_table_metadata("nuclei_v1")["description"])

# %% can be split by x y z position

nuclei_df = client.materialize.query_table("nuclei_v1", split_positions=True)
nuclei_df

# %% hierarchical annotations from schlegel et al paper
hierarchical_annos_df = client.materialize.query_table("hierarchical_neuron_annotations", limit=10)
hierarchical_annos_df

# %% or with subqueries/filtering
flow_annos_df = client.materialize.query_table("hierarchical_neuron_annotations", filter_equal_dict={"classification_system": "flow"})
flow_annos_df

cell_class_type_annos_df = client.materialize.query_table("hierarchical_neuron_annotations", filter_in_dict={"classification_system": ["cell_class", "cell_type"]})
cell_class_type_annos_df

# %% synapses 

''' Each synapse is a link from a pre- to a posynaptic site. 
As presynapses in the fly are usually polysynaptic, there are usually multiple 
synapses for each presynaptic site. They assigned a connection_score to every 
synapse which can be used to filter out false positives. Buhmann et al. suggest
 to correlate their predictions with those from Heinrich et al. who segmented 
 synaptic cleft on the same dataset. This is implemented through a cleft_score 
 that is associated with each synapse. We found that filtering synapses with 
 a cleft_score > 50 works well for removing false positives without using the 
 connection_score.

 Additionally, some synapses were annotated multiple times. We implemented 
 a distance based filtering to remove redundant synapses.

 We created a filtered view of the synapse table that automatically applies 
 all established filters to remove as many false positive synapses as possible. 
 It also adds neuropil information. The view published with version 630 is 
 valid_synapses_nt_np. For version 783 and later, we created valid_synapses_nt_np_v6,
   an updated version which only has minor differences to the previous version 
   (valid_synapses_nt_np will remain functional for consistency). 
   The main differences are that neuropil assignments in the optic lobes 
   were improved.
 '''

client.materialize.query_view("valid_synapses_nt_np_v6", limit=10)
#max per query is 500k, here example is limited to 10
#there are 110 million synapses in the full table


# %% synapses for a specific neuron

root_id = 720575940631693610

presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": [root_id]})
postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": [root_id]})

# %% example of further analysis: aanlyse super class distribution of synaptic partners

super_class_annos_df = client.materialize.query_table("hierarchical_neuron_annotations", filter_equal_dict={"classification_system": "super_class"})
unique_super_class = super_class_annos_df["cell_type"].unique()

pre_super_class_annos_df = super_class_annos_df[["pt_root_id", "cell_type"]].rename(
    columns={"pt_root_id": "pre_pt_root_id", "cell_type": "pre_super_class"})
post_super_class_annos_df = super_class_annos_df[["pt_root_id", "cell_type"]].rename(
    columns={"pt_root_id": "post_pt_root_id", "cell_type": "post_super_class"})

#get a specific neuron 
root_id = 720575940631693610
presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": [root_id]})
postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": [root_id]})

# merge synapse and annotation tables 
post_syn_super_df = pd.merge(postsyn_df, pre_super_class_annos_df, on="pre_pt_root_id")
pre_syn_super_df = pd.merge(presyn_df, post_super_class_annos_df, on="post_pt_root_id")

#plot 
plt.subplots(figsize=(12, 4))
sns.countplot(post_syn_super_df, x="pre_super_class", order=unique_super_class)
sns.countplot(pre_syn_super_df, x="post_super_class", order=unique_super_class)
# %%


root_id = 720575940631693610

#show all synapses post-synaptic to specified neuron
post_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": [root_id]})
n_postsynpartner=post_df.shape[0]
n_postneu =post_df['post_pt_root_id'].nunique()

#get all unique presynaptic sites on specified neuron
u_sites = post_df['pre_pt_supervoxel_id'].unique()

print(f"Number of postsynaptic connections: {n_postsynpartner} with {n_postneu} neurons \nNumber of unique presynaptic sites: {u_sites.shape[0]}")

#note: number of unique sites/postsynaptic connections is unexpectedly low
#note also: neurotransmitter prediction does not appear to be the same across all syanpses coming out of this neuron (not even close)


# %%

# %%
