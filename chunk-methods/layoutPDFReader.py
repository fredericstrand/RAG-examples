from llmsherpa.readers import LayoutPDFReader
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

reader = LayoutPDFReader("http://localhost:5010/api/parseDocument?renderFormat=all")
parsed = reader.read_pdf("/Users/fredericstrand/Documents/RAG-examples/graph-rag/data/lgsiv_avgjorelse_lg-1997-498.pdf")

docs = [Document(text=s.text, metadata={
    "title": s.title,
    "page_start": s.start_page_no,
    "page_end": s.end_page_no,
    "heading_hierarchy": s.heading_hierarchy,
}) for s in parsed.sections()]

index = VectorStoreIndex.from_documents(docs)
retriever = index.as_retriever(similarity_top_k=5)
for i, node in enumerate(retriever.retrieve("Gi et kort sammendrag"), 1):
    print(f"\nHit {i} (pages {node.metadata.get('page_start')}-{node.metadata.get('page_end')}):")
    print(node.text[:400])
