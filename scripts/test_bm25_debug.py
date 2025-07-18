#!/usr/bin/env python3
"""
BM25 로드 및 하이브리드 검색 디버깅 스크립트

단계별로 BM25를 로드하고 Vector Store와 결합해서 하이브리드 검색을 테스트합니다.
"""

import json
import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bm25_load():
    """BM25 인덱스 로드 테스트"""
    print("\n" + "="*60)
    print("🔍 Step 1: BM25 인덱스 로드 테스트")
    print("="*60)
    
    try:
        # 메타데이터 확인
        metadata_file = Path("data/bm25_index_langgraph/metadata.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"✅ 메타데이터 로드 성공:")
        print(f"   - 총 문서 수: {metadata['total_documents']}")
        print(f"   - 생성 시간: {metadata['created_at']}")
        
        # 문서들 로드 테스트
        docs_file = Path("data/bm25_index_langgraph/documents.json")
        with open(docs_file, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        print(f"✅ 문서 데이터 로드 성공: {len(docs_data)}개")
        
        # BM25 retriever 로드 시도
        try:
            retriever_file = Path("data/bm25_index_langgraph/bm25_retriever.pkl")
            with open(retriever_file, 'rb') as f:
                bm25_retriever = pickle.load(f)
            
            print("✅ BM25 retriever pickle 로드 성공")
            print(f"   - 타입: {type(bm25_retriever)}")
            
            return bm25_retriever, docs_data
            
        except Exception as e:
            print(f"❌ BM25 pickle 로드 실패: {e}")
            print("🔄 문서에서 BM25 재생성 시도...")
            
            # 문서에서 BM25 재생성
            from langchain_community.retrievers import BM25Retriever
            from langchain_core.documents import Document
            
            # 문서들을 Document 객체로 변환
            documents = []
            for doc_data in docs_data:
                doc = Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data["metadata"]
                )
                documents.append(doc)
            
            # BM25 재생성 (전처리 함수 없이)
            bm25_retriever = BM25Retriever.from_documents(documents)
            print("✅ BM25 retriever 재생성 성공")
            
            return bm25_retriever, docs_data
        
    except Exception as e:
        logger.error(f"BM25 로드 실패: {e}")
        return None, None

def test_bm25_search(bm25_retriever):
    """BM25 검색 테스트"""
    print("\n" + "="*60)
    print("🔍 Step 2: BM25 검색 테스트")
    print("="*60)
    
    test_queries = [
        "StateGraph workflow",
        "LangGraph tutorial",
        "checkpointing memory",
        "human in the loop",
        "graph nodes edges"
    ]
    
    for query in test_queries:
        print(f"\n📝 질의: {query}")
        print("-" * 40)
        
        try:
            results = bm25_retriever.get_relevant_documents(query, k=3)
            
            if results:
                for i, doc in enumerate(results, 1):
                    print(f"결과 {i}:")
                    print(f"  제목: {doc.metadata.get('title', 'N/A')}")
                    print(f"  내용: {doc.page_content[:100]}...")
            else:
                print("❌ 검색 결과 없음")
                
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
    
    return True

def test_vector_store():
    """Vector Store 로드 테스트"""
    print("\n" + "="*60)
    print("🔍 Step 3: Vector Store 로드 테스트")
    print("="*60)
    
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        # Embeddings 초기화
        embeddings = None
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            print("✅ OpenAI 임베딩 사용")
        except Exception as e:
            print(f"⚠️ OpenAI 임베딩 실패: {e}")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("✅ Google 임베딩 사용")
        
        # Vector Store 로드
        vectorstore = Chroma(
            collection_name="langgraph_docs",
            embedding_function=embeddings,
            persist_directory="./data/chroma_db_langgraph"
        )
        
        print("✅ Vector Store 로드 성공")
        
        # 간단한 검색 테스트
        results = vectorstore.similarity_search("StateGraph", k=2)
        print(f"✅ Vector 검색 테스트 성공: {len(results)}개 결과")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Vector Store 로드 실패: {e}")
        return None

def test_hybrid_search(vectorstore, bm25_retriever):
    """하이브리드 검색 테스트"""
    print("\n" + "="*60)
    print("🔍 Step 4: 하이브리드 검색 테스트")
    print("="*60)
    
    try:
        from langchain.retrievers import EnsembleRetriever
        
        # 앙상블 retriever 생성
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore.as_retriever(search_kwargs={"k": 3}), bm25_retriever],
            weights=[0.6, 0.4]  # Vector Store 60%, BM25 40%
        )
        
        print("✅ 하이브리드 retriever 생성 성공")
        
        # 테스트 질의
        test_query = "LangGraph StateGraph를 사용하여 워크플로우를 만들려면 어떻게 해야 하나요?"
        
        print(f"\n📝 테스트 질의: {test_query}")
        print("-" * 50)
        
        # Vector만 검색
        vector_results = vectorstore.similarity_search(test_query, k=3)
        print(f"\n🔍 Vector Store 결과 ({len(vector_results)}개):")
        for i, doc in enumerate(vector_results, 1):
            print(f"  {i}. {doc.metadata.get('title', 'N/A')[:50]}...")
        
        # BM25만 검색
        bm25_results = bm25_retriever.get_relevant_documents(test_query, k=3)
        print(f"\n🔍 BM25 결과 ({len(bm25_results)}개):")
        for i, doc in enumerate(bm25_results, 1):
            print(f"  {i}. {doc.metadata.get('title', 'N/A')[:50]}...")
        
        # 하이브리드 검색
        hybrid_results = ensemble_retriever.get_relevant_documents(test_query, k=5)
        print(f"\n🚀 하이브리드 결과 ({len(hybrid_results)}개):")
        for i, doc in enumerate(hybrid_results, 1):
            print(f"  {i}. {doc.metadata.get('title', 'N/A')[:50]}...")
            print(f"     내용: {doc.page_content[:80]}...")
        
        return ensemble_retriever
        
    except Exception as e:
        print(f"❌ 하이브리드 검색 실패: {e}")
        return None

def main():
    """메인 디버깅 함수"""
    print("🚀 BM25 + Vector Store 하이브리드 검색 디버깅 시작")
    
    # Step 1: BM25 로드
    bm25_retriever, docs_data = test_bm25_load()
    if not bm25_retriever:
        print("❌ BM25 로드 실패 - 종료")
        return False
    
    # Step 2: BM25 검색 테스트
    if not test_bm25_search(bm25_retriever):
        print("❌ BM25 검색 실패 - 종료")
        return False
    
    # Step 3: Vector Store 로드
    vectorstore = test_vector_store()
    if not vectorstore:
        print("❌ Vector Store 로드 실패 - 종료")
        return False
    
    # Step 4: 하이브리드 검색
    ensemble_retriever = test_hybrid_search(vectorstore, bm25_retriever)
    if not ensemble_retriever:
        print("❌ 하이브리드 검색 실패 - 종료")
        return False
    
    print("\n" + "="*60)
    print("🎉 모든 테스트 성공! 하이브리드 검색 준비 완료")
    print("="*60)
    
    return True

if __name__ == "__main__":
    main() 