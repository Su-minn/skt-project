#!/usr/bin/env python3
"""
Vector Store 검색 테스트 스크립트

구축된 Chroma Vector Store의 동작을 확인합니다.
"""

import os
import logging
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_search():
    """Vector Store 검색 테스트"""
    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        
        logger.info("Vector Store 검색 테스트 시작")
        
        # Embeddings 초기화
        embeddings = None
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            logger.info("OpenAI 임베딩 사용")
        except Exception as e:
            logger.warning(f"OpenAI 임베딩 실패: {e}")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            logger.info("Google 임베딩 사용")
        
        # Vector Store 로드
        vectorstore = Chroma(
            collection_name="langgraph_docs",
            embedding_function=embeddings,
            persist_directory="./data/chroma_db_langgraph"
        )
        
        # 테스트 질의들
        test_queries = [
            "LangGraph StateGraph를 사용하여 워크플로우를 만들려면 어떻게 해야 하나요?",
            "체크포인팅 기능은 어떻게 구현하나요?",
            "Human-in-the-loop는 무엇인가요?",
            "LangGraph에서 메모리 관리는 어떻게 하나요?",
            "StateGraph 노드를 정의하는 방법은?"
        ]
        
        print("\n" + "="*80)
        print("🔍 VECTOR STORE 검색 테스트 결과")
        print("="*80)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 테스트 {i}: {query}")
            print("-" * 60)
            
            try:
                # 유사도 검색 수행
                results = vectorstore.similarity_search_with_score(
                    query, 
                    k=3
                )
                
                if results:
                    for j, (doc, score) in enumerate(results, 1):
                        print(f"결과 {j} (유사도: {score:.3f}):")
                        print(f"제목: {doc.metadata.get('title', 'N/A')}")
                        print(f"URL: {doc.metadata.get('url', 'N/A')}")
                        print(f"내용: {doc.page_content[:200]}...")
                        print()
                else:
                    print("❌ 검색 결과 없음")
                    
            except Exception as e:
                print(f"❌ 검색 오류: {e}")
        
        # Vector Store 통계
        print("\n" + "="*50)
        print("📊 Vector Store 통계")
        print("="*50)
        
        # 컬렉션 정보 확인
        collection = vectorstore._collection
        doc_count = collection.count()
        print(f"총 문서 수: {doc_count}")
        
        # 샘플 메타데이터 확인
        sample_docs = vectorstore.similarity_search("LangGraph", k=1)
        if sample_docs:
            print(f"샘플 메타데이터: {sample_docs[0].metadata}")
        
        logger.info("Vector Store 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"Vector Store 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    test_vector_search() 