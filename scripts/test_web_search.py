#!/usr/bin/env python3
"""
Tavily Search 기반 Web Searcher 테스트 스크립트

새로 구현된 web_searcher 함수의 동작을 테스트합니다.
"""

import os
import logging
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_web_searcher():
    """Web Searcher 테스트"""
    try:
        # TAVILY_API_KEY 확인
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            print("❌ TAVILY_API_KEY 환경변수가 설정되지 않았습니다.")
            print("   .env 파일에 TAVILY_API_KEY=your_key 를 추가해주세요.")
            return False
        
        print("✅ TAVILY_API_KEY 확인 완료")
        
        # GraphState mock 객체 생성
        from knowledge_system.models.state import GraphState
        
        test_state = GraphState(
            query="LangGraph StateGraph를 사용하여 워크플로우를 만들려면 어떻게 해야 하나요?",
            current_step="testing"
        )
        
        # Web Searcher 함수 테스트
        from knowledge_system.nodes.searchers import web_searcher
        
        print("\n" + "="*60)
        print("🌐 Tavily Web Search 테스트 시작")
        print("="*60)
        print(f"질의: {test_state.query}")
        
        result = web_searcher(test_state)
        
        print(f"\n📊 검색 결과:")
        print(f"   - 결과 수: {len(result.get('web_results', []))}")
        print(f"   - 평균 관련성: {result.get('web_relevance_score', 0.0):.3f}")
        
        if 'error_message' in result:
            print(f"   - 오류: {result['error_message']}")
            return False
        
        # 개별 결과 출력
        for i, web_result in enumerate(result.get('web_results', []), 1):
            metadata = web_result.get('metadata', {})
            print(f"\n🔍 결과 {i}:")
            print(f"   제목: {metadata.get('title', 'N/A')}")
            print(f"   URL: {metadata.get('url', 'N/A')}")
            print(f"   도메인: {metadata.get('domain', 'N/A')}")
            print(f"   Tavily 점수: {metadata.get('tavily_score', 0.0):.3f}")
            print(f"   LLM 관련성: {metadata.get('relevance_score', 0.0):.3f}")
            print(f"   내용: {web_result.get('content', '')[:150]}...")
            print(f"   평가 이유: {metadata.get('relevance_reasoning', 'N/A')[:100]}...")
        
        print("\n" + "="*60)
        print("🎉 Tavily Web Search 테스트 성공!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False

def test_tavily_direct():
    """Tavily Search 직접 테스트"""
    try:
        from langchain_tavily import TavilySearch
        
        print("\n" + "="*60)
        print("🔍 Tavily Search 직접 테스트")
        print("="*60)
        
        search_web = TavilySearch(
            max_results=3,
            topic="general",
            search_depth="basic"
        )
        
        test_query = "LangGraph tutorial example"
        print(f"질의: {test_query}")
        
        result = search_web.invoke(test_query)
        
        print(f"\n📊 검색 결과:")
        print(f"   - 총 검색 결과: {result.get('total', 0)}")
        print(f"   - 반환된 결과: {len(result.get('results', []))}")
        print(f"   - 응답 시간: {result.get('response_time', 0):.2f}초")
        
        for i, item in enumerate(result.get('results', []), 1):
            print(f"\n📄 결과 {i}:")
            print(f"   제목: {item.get('title', 'N/A')}")
            print(f"   URL: {item.get('url', 'N/A')}")
            print(f"   점수: {item.get('score', 0.0):.3f}")
            print(f"   내용: {item.get('content', '')[:100]}...")
        
        print("\n✅ Tavily 직접 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ Tavily 직접 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Tavily Web Search 테스트 시작")
    
    # 1. Tavily 직접 테스트
    if not test_tavily_direct():
        print("❌ Tavily 직접 테스트 실패 - 종료")
        exit(1)
    
    # 2. Web Searcher 함수 테스트
    if not test_web_searcher():
        print("❌ Web Searcher 테스트 실패")
        exit(1)
    
    print("\n🎉 모든 테스트 성공! Tavily Web Search 준비 완료") 