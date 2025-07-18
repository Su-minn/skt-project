"""
LangGraph 워크플로우 구축

모든 노드와 엣지를 연결하여 계층적 검색 전략을 구현하는
LangGraph StateGraph를 생성합니다.
"""

import logging
from typing import Dict, Any
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from langgraph.graph import StateGraph, END, START

from knowledge_system.models.state import GraphState
from knowledge_system.nodes.analyzer import query_analyzer
from knowledge_system.nodes.searchers import graph_searcher, document_searcher, web_searcher
from knowledge_system.nodes.generator import answer_generator
from knowledge_system.edges.routing import route_to_next_search

logger = logging.getLogger(__name__)


def create_knowledge_graph():
    """
    LangGraph 지식 관리 시스템의 StateGraph를 생성합니다.
    
    워크플로우:
    START → QueryAnalyzer → [검색전략별 분기] → AnswerGenerator → END
    
    검색 전략:
    1. 그래프 검색 (개인 학습 자료) → 결과 평가 → 문서 검색 or 답변 생성
    2. 문서 검색 (공식 문서) → 결과 평가 → 웹 검색 or 답변 생성  
    3. 웹 검색 (최후 수단) → 답변 생성
    
    Returns:
        컴파일된 LangGraph StateGraph
    """
    logger.info("지식 관리 시스템 그래프 생성 시작")
    
    try:
        # StateGraph 인스턴스 생성
        workflow = StateGraph(GraphState)
        
        # === 노드 추가 ===
        logger.info("노드들 추가 중...")
        
        # 1. 질의 분석 노드
        workflow.add_node("query_analyzer", query_analyzer)
        
        # 2. 검색 노드들
        workflow.add_node("graph_searcher", graph_searcher)
        workflow.add_node("document_searcher", document_searcher)
        workflow.add_node("web_searcher", web_searcher)
        
        # 3. 답변 생성 노드  
        workflow.add_node("answer_generator", answer_generator)
        
        # === 엣지 연결 ===
        logger.info("엣지들 연결 중...")
        
        # 1. 시작점 → 질의 분석
        workflow.add_edge(START, "query_analyzer")
        
        # 2. 질의 분석 → 그래프 검색 (무조건)
        workflow.add_edge("query_analyzer", "graph_searcher")
        
        # 3. 그래프 검색 → 문서 검색 또는 답변 생성
        workflow.add_conditional_edges(
            "graph_searcher",
            route_to_next_search,
            {
                "document_searcher": "document_searcher",
                "answer_generator": "answer_generator"
            }
        )
        
        # 4. 문서 검색 → 웹 검색 또는 답변 생성
        workflow.add_conditional_edges(
            "document_searcher", 
            route_to_next_search,
            {
                "web_searcher": "web_searcher",
                "answer_generator": "answer_generator"
            }
        )
        
        # 5. 웹 검색 → 답변 생성 (무조건)
        workflow.add_edge("web_searcher", "answer_generator")
        
        # 6. 답변 생성 → 종료
        workflow.add_edge("answer_generator", END)
        
        # === 그래프 컴파일 ===
        logger.info("그래프 컴파일 중...")
        compiled_graph = workflow.compile()
        
        logger.info("지식 관리 시스템 그래프 생성 완료")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"그래프 생성 중 오류 발생: {str(e)}")
        raise


def create_simple_graph():
    """
    간단한 테스트용 그래프를 생성합니다.
    
    단순한 선형 워크플로우:
    START → QueryAnalyzer → GraphSearcher → AnswerGenerator → END
    
    Returns:
        컴파일된 간단한 StateGraph
    """
    logger.info("간단한 테스트 그래프 생성 시작")
    
    try:
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("query_analyzer", query_analyzer)
        workflow.add_node("graph_searcher", graph_searcher)
        workflow.add_node("answer_generator", answer_generator)
        
        # 선형 엣지 연결
        workflow.add_edge(START, "query_analyzer")
        workflow.add_edge("query_analyzer", "graph_searcher")
        workflow.add_edge("graph_searcher", "answer_generator")
        workflow.add_edge("answer_generator", END)
        
        compiled_graph = workflow.compile()
        
        logger.info("간단한 테스트 그래프 생성 완료")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"간단한 그래프 생성 중 오류 발생: {str(e)}")
        raise


def visualize_graph(graph, output_path: str = "knowledge_graph.png") -> None:
    """
    그래프를 시각화하여 파일로 저장합니다.
    
    Args:
        graph: 시각화할 CompiledGraph
        output_path: 저장할 파일 경로
    """
    try:
        # Mermaid 형태로 그래프 구조 출력
        print("\n=== LangGraph 워크플로우 구조 ===")
        print("graph TD")
        print("    START[사용자 질의] --> QA[질의 분석]")
        print("    QA --> |검색전략| GS[그래프 검색]")
        print("    QA --> |공식문서 우선| DS[문서 검색]")
        print("    QA --> |웹검색 우선| WS[웹 검색]")
        print("    GS --> |결과 우수| AG[답변 생성]")
        print("    GS --> |결과 부족| DS")
        print("    DS --> |결과 양호| AG")
        print("    DS --> |결과 부족| WS")
        print("    WS --> AG")
        print("    AG --> END[최종 답변]")
        print("===================================\n")
        
        logger.info(f"그래프 시각화 완료: {output_path}")
        
    except Exception as e:
        logger.warning(f"그래프 시각화 중 오류 발생: {str(e)}")


def get_graph_info(graph) -> Dict[str, Any]:
    """
    그래프의 기본 정보를 반환합니다.
    
    Args:
        graph: 정보를 조회할 CompiledGraph
        
    Returns:
        그래프 정보 딕셔너리
    """
    try:
        # 기본 그래프 정보 수집
        info = {
            "graph_type": "LangGraph StateGraph",
            "purpose": "계층적 지식 검색 시스템",
            "nodes": [
                "query_analyzer",
                "graph_searcher", 
                "document_searcher",
                "web_searcher",
                "answer_generator"
            ],
            "search_strategy": "개인자료 → 공식문서 → 웹검색",
            "state_model": "GraphState (Pydantic)",
            "routing_logic": "조건부 엣지 기반 지능형 라우팅"
        }
        
        return info
        
    except Exception as e:
        logger.error(f"그래프 정보 조회 중 오류 발생: {str(e)}")
        return {"error": str(e)}


# 그래프 빌더 팩토리 함수들
def build_development_graph():
    """개발용 그래프 (모든 기능 포함)"""
    return create_knowledge_graph()


def build_test_graph():
    """테스트용 그래프 (간단한 구조)"""
    return create_simple_graph()


def build_production_graph():
    """프로덕션용 그래프 (최적화된 버전)"""
    # 현재는 기본 그래프와 동일, 나중에 최적화 로직 추가
    return create_knowledge_graph() 