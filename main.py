import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_searcher():
    """Document searcher의 LLM 관련성 평가 테스트"""
    
    logger.info("=== Document Searcher 테스트 시작 ===")
    
    try:
        from knowledge_system.models.state import GraphState, QueryAnalysisResult
        from knowledge_system.nodes.searchers import document_searcher
        
        # 테스트 상태 생성
        analysis_result = QueryAnalysisResult(
            primary_entities=["LangGraph", "StateGraph"],
            intent_category="tutorial_request",
            context_hint="LangGraph StateGraph 사용법을 배우고 싶음"
        )
        
        state = GraphState(
            query="LangGraph StateGraph를 사용하여 워크플로우를 만들려면 어떻게 해야 하나요?",
            analysis_result=analysis_result
        )
        
        # document_searcher 실행
        result = document_searcher(state)
        
        # 결과 출력
        if "error_message" in result:
            logger.error(f"Document searcher 오류: {result['error_message']}")
        else:
            doc_results = result.get("doc_results", [])
            relevance_score = result.get("doc_relevance_score", 0.0)
            
            logger.info(f"문서 검색 결과 수: {len(doc_results)}")
            logger.info(f"LLM 평가 관련성 점수: {relevance_score:.2f}")
            
            for i, result_item in enumerate(doc_results, 1):
                logger.info(f"문서 결과 {i}:")
                logger.info(f"  소스: {result_item.get('source', 'Unknown')}")
                logger.info(f"  LLM 평가 점수: {result_item.get('relevance_score', 0.0)}")
                content = result_item.get('content', '')
                logger.info(f"  내용: {content[:100]}{'...' if len(content) > 100 else ''}")
                
                # 관련성 평가 이유 출력
                metadata = result_item.get('metadata', {})
                reasoning = metadata.get('relevance_reasoning', '이유 없음')
                logger.info(f"  LLM 평가 이유: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
        
        return result
        
    except Exception as e:
        logger.error(f"Document searcher 테스트 실패: {e}")
        return {"error": f"테스트 실패: {e}"}

def test_graph_searcher():
    """Neo4j GraphCypherQAChain을 사용한 graph_searcher 테스트"""
    
    # 환경 변수 확인
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Neo4j 환경 변수 누락: {missing_vars}")
        logger.info("Neo4j 연결 없이 기본 테스트 실행")
        return test_basic_workflow()
    
    logger.info("Neo4j 환경 변수 확인 완료 - Graph searcher 테스트 시작")
    
    try:
        from knowledge_system.models.state import GraphState, QueryAnalysisResult
        from knowledge_system.nodes.searchers import graph_searcher
        
        # 테스트 상태 생성
        analysis_result = QueryAnalysisResult(
            primary_entities=["LangGraph", "StateGraph"],
            intent_category="tutorial_request",
            context_hint="LangGraph StateGraph 사용법을 배우고 싶음"
        )
        
        state = GraphState(
            query="LangGraph StateGraph를 사용하여 워크플로우를 만들려면 어떻게 해야 하나요?",
            analysis_result=analysis_result
        )
        
        # graph_searcher 실행
        logger.info("=== Graph Searcher 테스트 시작 ===")
        result = graph_searcher(state)
        
        # 결과 출력
        if "error_message" in result:
            logger.error(f"Graph searcher 오류: {result['error_message']}")
        else:
            graph_results = result.get("graph_results", [])
            relevance_score = result.get("graph_relevance_score", 0.0)
            
            logger.info(f"검색 결과 수: {len(graph_results)}")
            logger.info(f"관련성 점수: {relevance_score:.2f}")
            
            for i, result_item in enumerate(graph_results, 1):
                logger.info(f"결과 {i}:")
                logger.info(f"  소스: {result_item.get('source', 'Unknown')}")
                logger.info(f"  점수: {result_item.get('relevance_score', 0.0)}")
                content = result_item.get('content', '')
                logger.info(f"  내용: {content[:100]}{'...' if len(content) > 100 else ''}")
        
        return result
        
    except ImportError as e:
        logger.error(f"라이브러리 import 실패: {e}")
        return {"error": f"Import 실패: {e}"}
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        return {"error": f"테스트 실패: {e}"}

def test_basic_workflow():
    """기본 워크플로우 테스트 (Neo4j 없이)"""
    logger.info("=== 기본 워크플로우 테스트 ===")
    
    try:
        from knowledge_system.models.state import GraphState, QueryAnalysisResult
        from knowledge_system.nodes.analyzer import query_analyzer
        
        # 1. 질의 분석 테스트
        state = GraphState(query="LangGraph StateGraph 사용법을 알려주세요")
        
        logger.info("1단계 - 질의 분석 시작")
        analysis_result = query_analyzer(state)
        
        if analysis_result.get("analysis_result"):
            logger.info("질의 분석 성공!")
            entities = analysis_result["analysis_result"].primary_entities
            intent = analysis_result["analysis_result"].intent_category
            logger.info(f"  엔티티: {entities}")
            logger.info(f"  의도: {intent}")
        else:
            logger.error("질의 분석 실패")
            
        return analysis_result
        
    except Exception as e:
        logger.error(f"기본 워크플로우 테스트 실패: {e}")
        return {"error": f"테스트 실패: {e}"}

def main():
    """메인 함수"""
    print("=== LangGraph Knowledge System 테스트 ===")
    
    # Document searcher의 LLM 관련성 평가 테스트
    logger.info("\n1. Document Searcher LLM 관련성 평가 테스트")
    doc_result = test_document_searcher()
    
    # Graph searcher 테스트 (Neo4j 가능한 경우)
    logger.info("\n2. Graph Searcher 테스트")
    graph_result = test_graph_searcher()
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()
