import os
from llama_index import GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, ServiceContext
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph

os.environ['OPENAI_API_KEY'] = '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'

index_path = './index/'
ls_indices = os.listdir(index_path)

index_set = {}
for doc in tqdm(ls_indices, total=len(ls_indices)):
    docname = doc[:-5]
    index = GPTSimpleVectorIndex.load_from_disk(index_path+doc)
    index_set[docname] = index

# describe each index to help traversal of composed graph
index_summaries = [f"paper by {p[6:-9]} written in {p[-9:-5]}" for p in ls_indices]

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[p[:-5]] for p in ls_indices],
    index_summaries=index_summaries,
    service_context=service_context,
)

# [optional] save to disk
graph.save_to_disk('./index/graph/indexgraph_xg_papers.json')

