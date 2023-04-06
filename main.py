import os
import pickle

from google.auth.transport.requests import Request

from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index import GPTSimpleVectorIndex, download_loader

os.environ['OPENAI_API_KEY'] = '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'

def authorize_gdocs():
    google_oauth2_scopes = [
        "https://www.googleapis.com/auth/documents.readonly"
    ]
    cred = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", 'rb') as token:
            cred = pickle.load(token)
    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            cred.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", google_oauth2_scopes)
            cred = flow.run_local_server(port=0)
        with open("token.pickle", 'wb') as token:
            pickle.dump(cred, token)

# function to authorize or download latest credentials
authorize_gdocs()

# initialize LlamaIndex google doc reader
GoogleDocsReader = download_loader('GoogleDocsReader')

# list of google docs we want to index
gdoc_ids = [
    '1DJZWjtyKpy9Y7Dp31T8SwKG5dU-RB2o8ZKG4LqpcAR4',
    '1REqdDqBA3mSWhEqKzWmV1w0azUz4yVvl8ljBPEdxnVQ'
    ]

loader = GoogleDocsReader()

# load gdocs and index them
documents = loader.load_data(document_ids=gdoc_ids)

index = GPTSimpleVectorIndex.from_documents(documents)

# Save your index to a index.json file
index.save_to_disk('index.json')
# Load the index from your saved index.json file
index = GPTSimpleVectorIndex.load_from_disk('index.json')

