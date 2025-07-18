"""
라우팅 로직

LangGraph 워크플로우에서 조건에 따라 다음 노드를 결정하는 
Edge 라우팅 함수들입니다.

계층적 검색 전략:
1. 개인 학습 자료 (그래프 검색) 우선
2. 공식 문서 (문서 검색) 보완  
3. 웹 검색 (최후 수단)
"""

import logging
from typing import Literal
from knowledge_system.models.state import GraphState

logger = logging.getLogger(__name__)

# 라우팅 결과 타입 정의
SearchRoute = Literal[
    "graph_searcher", 
    "document_searcher", 
    "web_searcher", 
    "answer_generator",
    "END"
]

NextStepRoute = Literal[
    "query_analyzer",
    "graph_searcher", 
    "document_searcher",
    "web_searcher",
    "answer_generator",
    "END"
]


def route_search_results(state: GraphState) -> SearchRoute:
    """
    검색 결과의 적합성에 따라 다음 단계를 결정하는 라우팅 함수
    
    라우팅 로직:
    1. 그래프 검색 결과가 충분히 좋으면 (0.8 이상) → 답변 생성
    2. 문서 검색 결과가 괜찮으면 (0.7 이상) → 답변 생성
    3. 웹 검색 결과가 있으면 (0.5 이상) → 답변 생성
    4. 아직 시도하지 않은 검색이 있으면 → 다음 검색 단계
    5. 모든 검색을 시도했으면 → 답변 생성 (최후)
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        다음에 실행할 노드 이름
    """
    logger.info("검색 결과 라우팅 시작")
    
    try:
        # 오류 상태인 경우 답변 생성으로 이동 (오류 메시지 포함)
        if state.error_message:
            logger.warning(f"오류 상태에서 답변 생성으로 라우팅: {state.error_message}")
            return "answer_generator"
        
        # 1. 그래프 검색 결과 확인
        if state.graph_relevance_score is not None:
            if state.graph_relevance_score >= 0.8:
                logger.info(f"그래프 검색 결과 우수 ({state.graph_relevance_score:.2f}) → 답변 생성")
                return "answer_generator"
        
        # 2. 문서 검색 결과 확인
        if state.doc_relevance_score is not None:
            if state.doc_relevance_score >= 0.5:
                logger.info(f"문서 검색 결과 양호 ({state.doc_relevance_score:.2f}) → 답변 생성")
                return "answer_generator"
        
        # 3. 웹 검색 결과 확인
        if state.web_relevance_score is not None:
            if state.web_relevance_score >= 0.5:
                logger.info(f"웹 검색 결과 사용 가능 ({state.web_relevance_score:.2f}) → 답변 생성")
                return "answer_generator"
        
        # 4. 아직 시도하지 않은 검색 단계 확인
        if state.graph_results is None:
            logger.info("그래프 검색 미실행 → 그래프 검색")
            return "graph_searcher"
        elif state.doc_results is None:
            logger.info("문서 검색 미실행 → 문서 검색")
            return "document_searcher"
        elif state.web_results is None:
            logger.info("웹 검색 미실행 → 웹 검색")
            return "web_searcher"
        
        # 5. 모든 검색을 시도했으면 답변 생성 (최후)
        logger.info("모든 검색 완료 → 답변 생성")
        return "answer_generator"
        
    except Exception as e:
        logger.error(f"라우팅 중 오류 발생: {str(e)}")
        return "answer_generator"  # 오류 발생시 답변 생성으로 진행


def route_to_next_search(state: GraphState) -> NextStepRoute:
    """
    현재 워크플로우 단계에 따라 다음 노드를 결정하는 라우팅 함수
    
    조건부 엣지 매핑과 정확히 일치하도록 수정됨:
    - graph_searcher에서 호출시: "document_searcher" 또는 "answer_generator"만 반환
    - document_searcher에서 호출시: "web_searcher" 또는 "answer_generator"만 반환
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        다음에 실행할 노드 이름
    """
    logger.info(f"워크플로우 라우팅 시작 - 현재 단계: {state.current_step}")
    
    try:
        current_step = state.current_step or "start"
        
        # 오류 상태인 경우 답변 생성으로 이동
        if current_step == "error" or state.error_message:
            logger.warning("오류 상태 → 답변 생성")
            return "answer_generator"
                
        elif current_step == "graph_searched":
            # 그래프 검색 결과에 따라 결정 (document_searcher 또는 answer_generator만)
            if state.graph_relevance_score and state.graph_relevance_score >= 0.8:
                logger.info("그래프 검색 결과 우수 → 답변 생성")
                return "answer_generator"
            else:
                logger.info("그래프 검색 결과 부족 → 문서 검색")
                return "document_searcher"
                
        elif current_step == "document_searched":
            # 문서 검색 결과에 따라 결정 (web_searcher 또는 answer_generator만)
            if state.doc_relevance_score and state.doc_relevance_score >= 0.5:
                logger.info("문서 검색 결과 양호 → 답변 생성")
                return "answer_generator"
            else:
                logger.info("문서 검색 결과 부족 → 웹 검색")
                return "web_searcher"
                
        elif current_step == "web_searched":
            # 웹 검색 후에는 무조건 답변 생성
            logger.info("웹 검색 완료 → 답변 생성")
            return "answer_generator"
            
        else:
            # 예상치 못한 단계이거나 완료 상태인 경우 답변 생성으로 이동
            logger.warning(f"알 수 없는 단계 '{current_step}' → 답변 생성")
            return "answer_generator"
            
    except Exception as e:
        logger.error(f"워크플로우 라우팅 중 오류 발생: {str(e)}")
        return "answer_generator"


def route_by_search_strategy(state: GraphState) -> NextStepRoute:
    """
    검색 전략에 따라 첫 번째 검색 노드를 결정하는 라우팅 함수
    
    검색 전략:
    - graph_first: 개인 학습 자료 우선 (기본)
    - document_first: 공식 문서 우선  
    - web_first: 웹 검색 우선
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        첫 번째 검색 노드 이름
    """
    # LangGraph 전용 시스템에서는 항상 그래프 우선 검색
    logger.info("검색 전략: 개인 자료 우선")
    return "graph_searcher"


def should_continue_search(state: GraphState) -> bool:
    """
    추가 검색이 필요한지 결정하는 헬퍼 함수
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        추가 검색 필요 여부
    """
    # 충분한 결과가 있으면 검색 중단
    if state.has_sufficient_results(threshold=0.7):
        return False
    
    # 모든 검색을 시도했으면 중단
    if (state.graph_results is not None and 
        state.doc_results is not None and 
        state.web_results is not None):
        return False
    
    # 그 외에는 계속 검색
    return True


def get_next_search_node(state: GraphState) -> NextStepRoute:
    """
    다음 검색 노드를 결정하는 헬퍼 함수
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        다음 검색 노드 이름
    """
    # 시도하지 않은 검색 중 우선순위가 높은 것부터
    if state.graph_results is None:
        return "graph_searcher"
    elif state.doc_results is None:
        return "document_searcher"
    elif state.web_results is None:
        return "web_searcher"
    else:
        return "answer_generator" 