#!/usr/bin/env python3
"""
하이브리드 검색 데이터베이스 구축 스크립트

크롤링된 LangGraph 공식문서를 Vector Store(Chroma)와 BM25 인덱스로 구축합니다.
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BM25용 전역 전처리 함수 (pickle 호환성을 위해)
def bm25_preprocess_func(text: str) -> List[str]:
    """BM25용 전처리 함수 (한국어 토크나이저)"""
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        
        # LangGraph 관련 전문 용어 추가
        kiwi.add_user_word('LangGraph', 'NNP')
        kiwi.add_user_word('StateGraph', 'NNP')
        kiwi.add_user_word('MessageGraph', 'NNP')
        kiwi.add_user_word('GraphState', 'NNP')
        kiwi.add_user_word('Checkpointer', 'NNP')
        kiwi.add_user_word('HITL', 'NNP')  # Human-in-the-loop
        
        tokens = kiwi.tokenize(text)
        return [token.form for token in tokens if len(token.form) > 1]
    except Exception:
        # fallback: 기본 분할
        return text.lower().split()

def load_crawled_docs(jsonl_file: str) -> List[Dict[str, Any]]:
    """크롤링된 문서들 로드"""
    docs = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                docs.append(doc)
        
        logger.info(f"문서 로드 완료: {len(docs)}개")
        return docs
    
    except Exception as e:
        logger.error(f"문서 로드 실패: {e}")
        return []

def create_document_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """문서를 청킹하여 검색 가능한 단위로 분할"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        
        # 텍스트 분할기 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # 청크 크기
            chunk_overlap=50,      # 청크 겹침
            length_function=len,
            separators=["\n\n", "\n", "## ", ". ", " ", ""]
        )
        
        all_chunks = []
        
        for doc_idx, doc in enumerate(docs):
            content = doc['content']
            title = doc['title']
            url = doc['url']
            
            # 문서를 청크로 분할
            chunks = text_splitter.split_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # 너무 짧은 청크는 제외
                    continue
                
                # Document 객체 생성
                chunk_doc = {
                    "page_content": chunk.strip(),
                    "metadata": {
                        "source": url,
                        "title": title,
                        "chunk_index": chunk_idx,
                        "doc_index": doc_idx,
                        "content_length": len(chunk),
                        "doc_type": "langgraph_official_doc",
                        "crawled_at": doc.get('crawled_at', ''),
                    }
                }
                all_chunks.append(chunk_doc)
        
        logger.info(f"청킹 완료: {len(all_chunks)}개 청크 생성")
        return all_chunks
        
    except Exception as e:
        logger.error(f"청킹 실패: {e}")
        return []

def build_vector_store(chunks: List[Dict[str, Any]]) -> str:
    """Vector Store (Chroma) 구축"""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_core.documents import Document
        
        # Document 객체로 변환
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
        
        # 임베딩 모델 초기화
        embeddings = None
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("OpenAI 임베딩 모델 초기화 성공")
        except Exception as e:
            logger.warning(f"OpenAI 임베딩 초기화 실패: {e}, Google 임베딩으로 대체")
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                logger.info("Google 임베딩 모델 초기화 성공")
            except Exception as e2:
                logger.error(f"임베딩 모델 초기화 완전 실패: {e2}")
                return None
        
        # Chroma 벡터 저장소 생성
        persist_directory = "./data/chroma_db_langgraph"
        
        # 기존 DB 삭제 (새로운 데이터로 재구축)
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
            logger.info("기존 벡터 DB 삭제 완료")
        
        # 새 벡터 DB 생성
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="langgraph_docs",
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector Store 생성 완료: {len(documents)}개 문서")
        logger.info(f"저장 위치: {persist_directory}")
        
        return persist_directory
        
    except Exception as e:
        logger.error(f"Vector Store 구축 실패: {e}")
        return None

def build_bm25_index(chunks: List[Dict[str, Any]]) -> str:
    """BM25 인덱스 구축 (한국어 토크나이저 포함)"""
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        import pickle
        
        # Document 객체로 변환
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
        
        # BM25 Retriever 생성
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=bm25_preprocess_func
        )
        
        # BM25 인덱스 저장 (문서들만 저장, retriever는 재생성)
        bm25_dir = Path("./data/bm25_index_langgraph")
        bm25_dir.mkdir(parents=True, exist_ok=True)
        
        # 문서들을 JSON으로 저장
        docs_file = bm25_dir / "documents.json"
        docs_data = []
        for doc in documents:
            docs_data.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        # 메타데이터 저장
        metadata = {
            "total_documents": len(documents),
            "preprocess_func": "bm25_preprocess_func",
            "created_at": "2025-07-18",
            "source": "LangGraph 공식문서"
        }
        
        metadata_file = bm25_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"BM25 인덱스 생성 완료: {len(documents)}개 문서")
        logger.info(f"저장 위치: {docs_file}")
        
        return str(docs_file)
        
    except Exception as e:
        logger.error(f"BM25 인덱스 구축 실패: {e}")
        return None

def test_hybrid_search(vector_db_path: str, bm25_path: str):
    """하이브리드 검색 테스트"""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        
        logger.info("하이브리드 검색 테스트 시작")
        
        # Vector Store 로드
        embeddings = None
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        vectorstore = Chroma(
            collection_name="langgraph_docs",
            embedding_function=embeddings,
            persist_directory=vector_db_path
        )
        
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # BM25 Retriever 재생성 (JSON에서 문서 로드)
        with open(bm25_path, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        documents = []
        for doc_data in docs_data:
            doc = Document(
                page_content=doc_data["page_content"],
                metadata=doc_data["metadata"]
            )
            documents.append(doc)
        
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=bm25_preprocess_func
        )
        
        # 앙상블 검색기 생성
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]  # Vector: 60%, BM25: 40%
        )
        
        # 테스트 질의
        test_queries = [
            "LangGraph StateGraph를 사용하여 워크플로우를 만들려면 어떻게 해야 하나요?",
            "persistence와 checkpointing에 대해 설명해주세요",
            "human in the loop 기능은 어떻게 작동하나요?"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 테스트 질의: {query}")
            
            try:
                results = ensemble_retriever.invoke(query)
                logger.info(f"검색 결과: {len(results)}개")
                
                for i, doc in enumerate(results[:2]):  # 상위 2개만 출력
                    logger.info(f"[{i+1}] 제목: {doc.metadata.get('title', 'N/A')}")
                    logger.info(f"    내용: {doc.page_content[:100]}...")
                    
            except Exception as e:
                logger.error(f"검색 실패: {e}")
        
        logger.info("✅ 하이브리드 검색 테스트 완료")
        
    except Exception as e:
        logger.error(f"하이브리드 검색 테스트 실패: {e}")


def main():
    """메인 실행 함수"""
    print("🚀 하이브리드 검색 데이터베이스 구축 시작")
    print("=" * 50)
    
    # 1. 크롤링된 문서 로드
    jsonl_file = "data/langgraph_docs/langgraph_docs.jsonl"
    if not os.path.exists(jsonl_file):
        print(f"❌ 크롤링된 문서를 찾을 수 없습니다: {jsonl_file}")
        print("먼저 scripts/langgraph_docs_crawler.py를 실행하세요.")
        return
    
    docs = load_crawled_docs(jsonl_file)
    if not docs:
        print("❌ 문서 로드 실패")
        return
    
    # 2. 문서 청킹
    print("\n📝 문서 청킹 중...")
    chunks = create_document_chunks(docs)
    if not chunks:
        print("❌ 청킹 실패")
        return
    
    # 3. Vector Store 구축
    print("\n🔤 Vector Store 구축 중...")
    vector_db_path = build_vector_store(chunks)
    if not vector_db_path:
        print("❌ Vector Store 구축 실패")
        return
    
    # 4. BM25 인덱스 구축
    print("\n🔍 BM25 인덱스 구축 중...")
    bm25_path = build_bm25_index(chunks)
    if not bm25_path:
        print("❌ BM25 인덱스 구축 실패")
        return
    
    # 5. 하이브리드 검색 테스트
    print("\n🧪 하이브리드 검색 테스트 중...")
    test_hybrid_search(vector_db_path, bm25_path)
    
    print("\n✅ 하이브리드 검색 데이터베이스 구축 완료!")
    print(f"📊 처리된 청크: {len(chunks)}개")
    print(f"🔤 Vector Store: {vector_db_path}")
    print(f"🔍 BM25 Index: {bm25_path}")
    print("\n💡 이제 knowledge_system/nodes/searchers.py의 document_searcher를 업데이트할 수 있습니다!")


if __name__ == "__main__":
    main() 