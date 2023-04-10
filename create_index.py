import os
from llama_index import download_loader, GPTSimpleVectorIndex, ServiceContext, SimpleDirectoryReader, GPTListIndex
from tqdm import tqdm

PDFReader = download_loader("PDFReader")
loader = PDFReader()

os.environ['OPENAI_API_KEY'] = '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'

doc_path = '/home/max/sciebo/local/dshs/PhD/phd_Literatur/sports_analytics/XG/'
ls_papers = os.listdir(doc_path)

service_context = ServiceContext.from_defaults(chunk_size_limit=512)
index_set = {}
for doc in tqdm(ls_papers, total=len(ls_papers)):
    docname = doc[:-4]
    document = loader.load_data(doc_path+doc)
    #document = SimpleDirectoryReader(doc_path+doc).load_data()
    cur_index = GPTSimpleVectorIndex.from_documents(document, service_context=service_context)
    index_set[docname] = cur_index
    cur_index.save_to_disk(f'./index/index_{docname}.json')
