### 分块要考虑语义完整性
把文档切得太碎不一定是好事。如果分块把定义、表格、列表这些结构性内容拦腰切断，模型拿到的上下文就是残缺的。

所以要用句子级别的分割器，设置合理的重叠区间，保证每个节点语义完整。

from llama_index.core.node_parser import SentenceSplitter  
from llama_index.core import Settings  

# Good default for most tech docs  
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)  
nodes = splitter.get_nodes_from_documents(documents)  

# Embed + index as usual  
Settings.chunk_size = 512  # keep aligned
比如代码文档和API说明用384-768 tokens，重叠比例10-15%左右。长篇PDF文档可以放宽到768-1024，overlap设64基本够用，能保持引用关系不断。


2) 扩展命中句子的上下文窗口
单独一句话作为检索结果，用户看着没把握。LlamaIndex的sentence-window后处理器会在命中句子前后各补几句，让返回的片段更连贯，也更方便直接引用。

from llama_index.postprocessor.sentence_window import SentenceWindowNodePostprocessor  

post = SentenceWindowNodePostprocessor(  
    window_size=2,  # grab ±2 sentences  
    original_text=True  
)  

retriever = index.as_retriever(similarity_top_k=8, node_postprocessors=[post])
适用场景： FAQ、规章制度、会议纪要这类文档特别有用，往往一句话的差异就能改变整个答案的意思。

3) 向量检索配合BM25做混合召回
向量embedding擅长捕捉语义相似度，BM25擅长精确匹配关键词。两个方法各有偏向，融合起来用效果明显好于单一方案。

from llama_index.retrievers.bm25 import BM25Retriever  
from llama_index.core.retrievers import VectorIndexRetriever  
from llama_index.retrievers.fusion import QueryFusionRetriever  

bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)  
vec  = VectorIndexRetriever(index=index, similarity_top_k=10)  

hybrid = QueryFusionRetriever(  
    retrievers=[vec, bm25],  
    similarity_top_k=10,       # final top-k after fusion  
    mode="reciprocal_rerank"   # simple, strong baseline  
)
内部知识库测试，混合检索+rerank（下一条会说）相比纯向量检索，correct@1指标能提升10-18%左右。

4) 用cross-encoder做二次排序
检索阶段只是粗筛候选，真正决定哪些结果靠谱还得靠reranker。一个轻量的cross-encoder模型（比如MiniLM）能有效把那些看起来相似但实际答非所问的结果往后排。

from llama_index.postprocessor import SentenceTransformerRerank  

rerank = SentenceTransformerRerank(  
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",  
    top_n=5  
)  

retriever = index.as_retriever(similarity_top_k=20, node_postprocessors=[rerank])
cross-encoder会同时看query和passage，不是简单的向量相似度比对而是直接判断段落是否真的在回答问题。


5) 查询重写：HyDE和多query生成
用户的问题经常描述不清楚。HyDE (Hypothetical Document Embeddings) 的思路是让LLM先生成一个理想答案，然后用这个假设答案去embedding和检索，能减少query本身表达不清造成的召回偏差。Multi-query则是生成多个改写版本，从不同角度去检索。

from llama_index.core.query_engine import RetrieverQueryEngine  
from llama_index.core.query_transform.base import HyDEQueryTransform  
from llama_index.core.query_transform import MultiQueryTransform  

base_retriever = index.as_retriever(similarity_top_k=12)  

hyde = HyDEQueryTransform()                # creates hypothetical answer  
multi = MultiQueryTransform(num_queries=4) # diverse paraphrases  

engine = RetrieverQueryEngine.from_args(  
    retriever=base_retriever,  
    query_transform=[hyde, multi]          # chain transforms  
)
例如那种模糊的业务问题，像”怎么回滚计费系统”，或者行业术语有歧义的领域，这个方法效果不错。


6) 元数据过滤优先于向量检索
不是所有文档都需要参与检索。如果能提前用metadata把范围限定在特定产品、版本、团队、时间段，检索效率和准确性都会高很多。

from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter, FilterCondition  
from llama_index.core.retrievers import VectorIndexRetriever  

filters = MetadataFilters(  
    filters=[ExactMatchFilter(key="product", value="billing"),  
             ExactMatchFilter(key="version", value="v3")],  
    condition=FilterCondition.AND  
)  

retriever = VectorIndexRetriever(  
    index=index,  
    similarity_top_k=12,  
    filters=filters  
)
对于政策文档和release notes，可以在后处理阶段加个时间过滤，防止过时信息混进来。


7) 压缩冗余上下文
长context不仅费token，还会引入噪声。Contextual compression能把检索到的内容再过滤一遍，只留下跟当前问题强相关的部分。

from llama_index.retrievers import ContextualCompressionRetriever  
from llama_index.postprocessor import LLMRerank  

base = index.as_retriever(similarity_top_k=12)  

# Use an LLM-powered  deephub relevance filter or a local mini model  
compressor = LLMRerank(top_n=6)   

compressed_retriever = ContextualCompressionRetriever(  
    base_retriever=base,  
    compressor=compressor  
)
这样prompt变短，答案也更聚焦，成本降下来了，这种方式对于处理那种章节很长的PDF特别管用。


8) 动态调整top_k：基于相似度阈值
写死top_k其实挺粗的。如果检索出来的结果分数都很低，硬塞进prompt反而干扰模型；如果前几个结果分数已经很高，后面的也没必要全要。

from llama_index.postprocessor import SimilarityPostprocessor  

cut = SimilarityPostprocessor(similarity_cutoff=0.78)  

retriever = index.as_retriever(similarity_top_k=20, node_postprocessors=[cut])  

# Optional: shrink k dynamically deep hub based on score slope  
def dynamic_k(scores, min_k=3, max_k=12, drop=0.08):  
    # when score difference flattens, stop  
    k = min_k  
    for i in range(1, min(len(scores), max_k)):  
        if scores[i-1] - scores[i] > drop:  
            k = i+1  
    return max(min_k, min(k, max_k))
把”总是塞进一堆垃圾”变成”只要有用的部分”,单靠这一条改动，幻觉率就能明显下降。


组合使用的一个完整pipeline示例
实际项目里这些技巧通常会组合用：

数据准备阶段： 用sentence-aware splitter切分文档（512/64），同时给每个chunk打上metadata（product、version、date、author等）

检索阶段： 混合检索（vector + BM25）→ 相似度阈值过滤 → cross-encoder重排序

Query处理： 对模糊问题用HyDE + multi-query做改写

结果精简： 用LLM或embedding rerank压缩到4-8个高相关句子

生成答案： 把精简后的context喂给LLM，记得标注引用来源

实测：某个2400份文档的内部知识库，上了混合检索 + rerank + 元数据过滤这套组合，correct@1从63%涨到78%，”我不确定”这类回答少了30%左右。具体效果会因数据而异，但宽召回、严过滤、精确引用这个思路基本适用。

一个可运行的代码示例

from llama_index.core import VectorStoreIndex  
from llama_index.retrievers.bm25 import BM25Retriever  
from llama_index.retrievers.fusion import QueryFusionRetriever  
from llama_index.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor  
from llama_index.postprocessor.sentence_window import SentenceWindowNodePostprocessor  
from llama_index.core.query_engine import RetrieverQueryEngine  
from llama_index.core.query_transform.base import HyDEQueryTransform  
from llama_index.core.query_transform import MultiQueryTransform  

# Build your index deep elsewhere; assume `index`  hub exists.  
bm25 = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)  
vec  = index.as_retriever(similarity_top_k=12)  

hybrid = QueryFusionRetriever(  
    retrievers=[vec, bm25],  
    similarity_top_k=16,  
    mode="reciprocal_rerank"  
)  

post = [  
    SimilarityPostprocessor(similarity_cutoff=0.78),  
    SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=6),  
    SentenceWindowNodePostprocessor(window_size=2, original_text=True),  
]  

engine = RetrieverQueryEngine.from_args(  
    retriever=hybrid,  
    node_postprocessors=post,  
    query_transform=[HyDEQueryTransform(), MultiQueryTransform(num_queries=4)]  
)  

resp = engine.query("How do we safely roll back billing in v3?")  
print(resp)
评估指标
衡量检索好坏真正有用的指标就这几个：

Correct@1 — 人工评判或抽样标注。全量标注不现实的话，每个迭代周期标50个query就够看出趋势

可引用性 — 平均context长度和引用片段数。越短越清晰的引用，用户信任度越高

延迟预算 — p95首token时间。压缩带来的收益通常能cover掉rerank的开销

安全机制 — 当top结果分数都低于阈值时，明确返回”未找到可靠信息”，别硬答

RAG质量其实就是工程问题。几个靠谱的默认配置加上精细的后处理，效果就能上一个台阶。

总结
先从chunking和sentence window入手，这两个改动成本最低。然后加混合检索和cross-encoder，这是性价比最高的组合。后面再根据实际问题针对性地补充HyDE/multi-query（解决query不清晰）、metadata filter（限定范围）、compression（降噪）、adaptive k（提升置信度）。第一次demo就能看出明显差异。

作者：Hash Block
