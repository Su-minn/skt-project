"""
LLM 기반 질의 분석 노드

사용자 질의를 LLM으로 분석하여 구조화된 정보를 추출합니다.
Structured Output을 사용하여 LangGraph 개인 지식 검색에 최적화된 정보를 제공합니다.
"""

import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from knowledge_system.models.state import GraphState, QueryAnalysisResult

logger = logging.getLogger(__name__)


def get_llm_instance():
    """
    환경변수에 따라 적절한 LLM 인스턴스를 반환합니다.
    
    우선순위: OpenAI -> Google Gemini -> Mock LLM (테스트용)
    """
    # OpenAI API 키 확인
    if os.getenv("OPENAI_API_KEY"):
        logger.info("OpenAI LLM 사용")
        return ChatOpenAI(
            model="gpt-4.1",  # 비용 효율적인 모델
            temperature=0.1,      # 일관성을 위해 낮은 온도
        )
    
    # Google Gemini API 키 확인  
    elif os.getenv("GOOGLE_API_KEY"):
        logger.info("Google Gemini LLM 사용")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # 빠르고 비용 효율적
            temperature=0.1,
        )
    
    else:
        logger.warning("LLM API 키가 설정되지 않음. Mock 분석 사용")
        return None


def create_analysis_prompt():
    """질의 분석을 위한 프롬프트 템플릿을 생성합니다."""
    
    system_message = """당신은 LangGraph 관련 사용자 질의를 분석하는 전문가입니다.
질의를 분석하여 개인 지식 그래프 검색 시스템이 최적의 검색을 수행할 수 있도록 구조화된 정보를 추출해주세요.

주요 분석 대상:
- LangGraph의 StateGraph, Node, Edge 등 핵심 개념
- React Agent, Multi-Agent 등 에이전트 패턴
- 워크플로우, 라우팅, 상태 관리 등 구현 기법
- 문제 해결, 코드 생성, 설명 요청 등 사용자 의도

분석할 요소들:
1. **핵심 개념/엔티티**: 사용자 질의에서 실제로 언급된 LangGraph 관련 주요 기술, 도구, 개념만 추출
   - 질의에 명시적으로 언급된 개념만 포함
   - 추측이나 일반적인 개념은 제외
   - 영어/한글 키워드 모두 고려
2. **의도 파악**: 사용자가 원하는 것 (정보검색, 코드생성, 설명, 문제해결 등)
3. **컨텍스트 힌트**: 검색과 답변 생성에 도움이 될 추가 정보

중요: primary_entities에는 사용자가 실제로 질의에서 언급한 개념만 포함해야 합니다."""

    user_template = """사용자 질의: {query}

위 LangGraph 관련 질의를 분석하여 구조화된 정보를 추출해주세요.
특히 primary_entities에는 질의에서 실제로 언급된 개념만 포함해주세요."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_template)
    ])


def mock_query_analysis(query: str) -> QueryAnalysisResult:
    """
    API 키가 없을 때 사용하는 Mock 분석 함수
    
    기본적인 키워드 기반 분석을 수행합니다.
    """
    query_lower = query.lower()
    
    # 실제 질의에서 언급된 엔티티만 추출
    primary_entities = []
    
    # 질의에서 실제로 언급된 LangGraph 관련 키워드만 추출
    langgraph_keywords = {
        "langgraph": "LangGraph",
        "stategraph": "StateGraph", 
        "node": "Node",
        "edge": "Edge", 
        "react": "React",
        "agent": "Agent",
        "workflow": "Workflow",
        "routing": "Routing",
        "memory": "Memory",
        "state": "State",
        "graph": "Graph",
        "subgraph": "SubGraph",
        "multi-agent": "Multi-Agent",
        "streaming": "Streaming",
        "parallel": "Parallel",
        "병렬": "Parallel",
        "실행": "Execution",
        "개념": "Concept",
        "예시": "Example"
    }
    
    # 질의에서 실제로 언급된 키워드만 찾기
    for keyword, entity in langgraph_keywords.items():
        if keyword in query_lower:
            primary_entities.append(entity)
    
    # 의도 분류
    if any(word in query_lower for word in ["만들", "구현", "생성", "코드", "작성", "개발"]):
        intent = "code_generation"
    elif any(word in query_lower for word in ["설명", "어떻게", "왜", "무엇", "차이", "정의"]):
        intent = "explanation_request"
    elif any(word in query_lower for word in ["문제", "오류", "에러", "안됨", "해결", "디버그"]):
        intent = "troubleshooting"
    elif any(word in query_lower for word in ["비교", "차이점", "versus", "vs", "대비"]):
        intent = "comparison"
    elif any(word in query_lower for word in ["튜토리얼", "가이드", "방법", "순서", "단계"]):
        intent = "tutorial_request"
    elif any(word in query_lower for word in ["모범", "베스트", "best", "practice", "권장"]):
        intent = "best_practices"
    else:
        intent = "information_search"
    
    # 컨텍스트 힌트 생성
    if primary_entities:
        context_hint = f"LangGraph 관련 {intent} 질의. 핵심 개념: {', '.join(primary_entities)}"
    else:
        context_hint = f"LangGraph 관련 {intent} 질의"
    
    return QueryAnalysisResult(
        primary_entities=primary_entities,
        intent_category=intent,
        context_hint=context_hint
    )


def query_analyzer(state: GraphState) -> Dict[str, Any]:
    """
    LLM 기반 사용자 질의 분석 노드
    
    Structured Output을 사용하여 질의를 분석하고
    LangGraph 개인 지식 검색에 최적화된 구조화된 정보를 추출합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 정보
    """
    logger.info(f"LLM 기반 질의 분석 시작: {state.query}")
    
    try:
        # LLM 인스턴스 가져오기
        llm = get_llm_instance()
        
        if llm is None:
            # Mock 분석 사용
            logger.info("Mock 분석 사용")
            analysis_result = mock_query_analysis(state.query)
        else:
            # LLM 기반 구조화된 분석
            logger.info("LLM 기반 구조화된 분석 수행")
            
            # Structured Output LLM 생성
            structured_llm = llm.with_structured_output(QueryAnalysisResult)
            
            # 프롬프트 생성
            prompt = create_analysis_prompt()
            
            # 분석 체인 생성 및 실행
            analysis_chain = prompt | structured_llm
            analysis_result = analysis_chain.invoke({"query": state.query})
        
        logger.info(f"분석 완료 - 핵심 엔티티: {analysis_result.primary_entities}")
        logger.info(f"의도: {analysis_result.intent_category}")
        logger.info(f"컨텍스트: {analysis_result.context_hint}")
        
        return {
            "analysis_result": analysis_result,
            "current_step": "query_analyzed"
        }
        
    except Exception as e:
        logger.error(f"LLM 질의 분석 중 오류 발생: {str(e)}")
        logger.info("오류 발생, Mock 분석으로 대체")
        
        # 오류 시 Mock 분석으로 대체
        try:
            analysis_result = mock_query_analysis(state.query)
            return {
                "analysis_result": analysis_result,
                "current_step": "query_analyzed"
            }
        except Exception as fallback_error:
            logger.error(f"Mock 분석도 실패: {str(fallback_error)}")
            return {
                "error_message": f"질의 분석 실패: {str(e)}",
                "current_step": "error"
            } 