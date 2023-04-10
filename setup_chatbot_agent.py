import os
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.indices.composability import ComposableGraph
from llama_index import LLMPredictor, ServiceContext, GPTSimpleVectorIndex
from langchain import OpenAI
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, GraphToolConfig
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
os.environ['OPENAI_API_KEY'] = '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'

index_path = './index/'
ls_indices = os.listdir(index_path)

index_set = {}
for doc in ls_indices:
    if 'json' in doc:
        docname = doc[:-5]
        index = GPTSimpleVectorIndex.load_from_disk(index_path+doc)
        index_set[docname] = index

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
graph = ComposableGraph.load_from_disk(
    './index/graph/indexgraph_xg_papers.json',
    service_context=service_context,
)


decompose_transform = DecomposeQueryTransform(
    llm_predictor, verbose=True
)

# define query configs for graph
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1,
            # "include_summary": True
        },
        "query_transform": decompose_transform
    },
    {
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        }
    },
]
# graph config
graph_config = GraphToolConfig(
    graph=graph,
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple papers.",
    query_configs=query_configs,
    tool_kwargs={"return_direct": True}
)

# define toolkit
index_configs = []
ls_papers = [p[:-5] for p in ls_indices if 'json' in p]
for p in ls_papers:
    tool_config = IndexToolConfig(
        index=index_set[p],
        name=f"Vector Index {p}",
        description=f"useful for when you want to answer queries about the {p} paper",
        index_query_kwargs={"similarity_top_k": 3},
        tool_kwargs={"return_direct": True}
    )
    index_configs.append(tool_config)

toolkit = LlamaToolkit(
    index_configs=index_configs,
    graph_configs=[graph_config]
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)

if __name__ == "__main__":
    while True:
        text_input = input("User: ")
        response = agent_chain.run(input=text_input)
        print(f'Agent: {response}')

# TODO: set up as proper software
# force chatbot to query all vector indices
# other configurations of langchain and llamaindex
# other textcoropra
# automate paper ingestion
# try out arxive