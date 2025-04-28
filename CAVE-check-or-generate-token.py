from caveclient import CAVEclient
client = CAVEclient()
auth = client.auth
print(auth.token)


#get new token, this gives instructions on how do do it 

#auth.get_new_token()
