from caveclient import CAVEclient


client = CAVEclient(
    datastack_name="prieto_godino_fly_larva",
    server_address="https://proofreading.zetta.ai"
)
auth = client.auth()

auth.get_new_token()

