"""
답변 생성 노드

검색된 정보들을 종합하여 사용자 질의에 대한 
최적의 답변을 생성하는 노드입니다.
"""

import logging
from typing import Dict, Any, List
from knowledge_system.models.state import GraphState, SearchResult

logger = logging.getLogger(__name__)


def answer_generator(state: GraphState) -> Dict[str, Any]:
    """
    최종 답변 생성 노드
    
    모든 검색 결과를 종합하여 사용자 질의에 대한 
    정확하고 유용한 답변을 생성합니다.
    
    답변 생성 전략:
    1. 개인 학습 자료 우선 활용
    2. 공식 문서로 보완
    3. 웹 검색 결과로 최신 정보 추가
    4. 소스 출처 명시
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 정보 (final_answer, sources, confidence_score)
    """
    logger.info("답변 생성 시작")
    
    try:
        query = state.query
        
        # analysis_result에서 직접 정보 추출
        intent = "information_search"  # 기본값
        if state.analysis_result:
            intent = state.analysis_result.intent_category
        
        # 최고 품질의 검색 결과 수집
        best_results = state.get_best_results(max_results=5)
        
        if not best_results:
            # 검색 결과가 없는 경우
            return {
                "final_answer": f"죄송합니다. '{query}'에 대한 관련 정보를 찾을 수 없습니다. 질의를 다시 구체화해 주시기 바랍니다.",
                "sources": [],
                "confidence_score": 0.1,
                "current_step": "answer_generated"
            }
        
        # LLM 초기화
        llm = None
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
            logger.info("OpenAI LLM 사용")
        except Exception as e:
            logger.warning(f"OpenAI LLM 실패: {e}")
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
                logger.info("Google LLM으로 대체")
            except Exception as e2:
                logger.error(f"LLM 초기화 실패: {e2}")
                # LLM 실패시 템플릿 기반 답변 생성
                context_parts = []
                sources = []
                for result in best_results:
                    context_parts.append(f"[{result['source']}] {result['content']}")
                    sources.append(result['source'])
                return generate_template_answer(query, context_parts, sources, intent)
        
        # 검색 결과를 컨텍스트로 구성
        context_parts = []
        sources = []
        
        for result in best_results:
            context_parts.append(f"[{result['source']}] {result['content']}")
            sources.append(result['source'])
        
        # LLM 기반 답변 생성
        answer = generate_llm_answer(query, context_parts, sources, intent, llm)
        
        # 신뢰도 점수 계산
        confidence_score = calculate_confidence_score(best_results, state)
        
        logger.info(f"답변 생성 완료 - 신뢰도: {confidence_score}")
        
        return {
            "final_answer": answer,
            "sources": sources,
            "confidence_score": confidence_score,
            "current_step": "completed"
        }
        
    except Exception as e:
        logger.error(f"답변 생성 중 오류 발생: {str(e)}")
        return {
            "error_message": f"답변 생성 실패: {str(e)}",
            "final_answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
            "sources": [],
            "confidence_score": 0.0,
            "current_step": "error"
        }


def generate_code_answer(query: str, context_parts: List[str], sources: List[str]) -> str:
    """코드 생성 요청에 대한 답변 생성"""
    
    answer = f"**{query}에 대한 코드 예제:**\n\n"
    
    # 컨텍스트에서 코드 관련 정보 추출 (간단한 예시)
    for context in context_parts:
        if any(keyword in context.lower() for keyword in ["코드", "예제", "구현", "class", "def", "import"]):
            answer += f"{context}\n\n"
    
    answer += "**참고 자료:**\n"
    for i, source in enumerate(sources, 1):
        answer += f"{i}. {source}\n"
    
    return answer


def generate_explanation_answer(query: str, context_parts: List[str], sources: List[str]) -> str:
    """설명 요청에 대한 답변 생성"""
    
    answer = f"**{query}에 대한 설명:**\n\n"
    
    # 설명 구성
    for context in context_parts:
        answer += f"• {context}\n\n"
    
    answer += "**참고 자료:**\n"
    for i, source in enumerate(sources, 1):
        answer += f"{i}. {source}\n"
    
    return answer


def generate_troubleshooting_answer(query: str, context_parts: List[str], sources: List[str]) -> str:
    """문제 해결에 대한 답변 생성"""
    
    answer = f"**{query} 문제 해결 방법:**\n\n"
    
    answer += "**관련 정보:**\n"
    for context in context_parts:
        answer += f"• {context}\n\n"
    
    answer += "**해결 단계:**\n"
    answer += "1. 문제 상황을 정확히 파악하세요\n"
    answer += "2. 관련 문서를 참조하세요\n"
    answer += "3. 단계별로 접근해보세요\n\n"
    
    answer += "**참고 자료:**\n"
    for i, source in enumerate(sources, 1):
        answer += f"{i}. {source}\n"
    
    return answer


def generate_information_answer(query: str, context_parts: List[str], sources: List[str]) -> str:
    """일반 정보 요청에 대한 답변 생성"""
    
    answer = f"**{query}에 대한 정보:**\n\n"
    
    for context in context_parts:
        answer += f"{context}\n\n"
    
    answer += "**참고 자료:**\n"
    for i, source in enumerate(sources, 1):
        answer += f"{i}. {source}\n"
    
    return answer


def generate_llm_answer(query: str, context_parts: List[str], sources: List[str], intent: str, llm) -> str:
    """LLM을 사용한 답변 생성"""
    
    # 컨텍스트 구성
    context = "\n\n".join(context_parts)
    
    # 의도별 프롬프트 구성
    if intent == "code_generation":
        system_prompt = """당신은 프로그래밍 전문가입니다. 주어진 컨텍스트를 바탕으로 사용자의 질의에 대한 코드 예제를 생성해주세요.

답변 형식:
1. 코드 예제 (실행 가능한 형태)
2. 코드 설명
3. 참고 자료

코드는 실제로 실행 가능해야 하며, 주어진 컨텍스트의 정보를 최대한 활용하세요."""
    
    elif intent == "explanation_request":
        system_prompt = """당신은 기술 전문가입니다. 주어진 컨텍스트를 바탕으로 사용자의 질의에 대한 명확하고 이해하기 쉬운 설명을 제공해주세요.

답변 형식:
1. 핵심 개념 설명
2. 상세 설명
3. 참고 자료

설명은 체계적이고 논리적이어야 하며, 주어진 컨텍스트의 정보를 기반으로 해야 합니다."""
    
    elif intent == "troubleshooting":
        system_prompt = """당신은 문제 해결 전문가입니다. 주어진 컨텍스트를 바탕으로 사용자의 문제에 대한 해결 방법을 제시해주세요.

답변 형식:
1. 문제 분석
2. 해결 단계
3. 예방 방법
4. 참고 자료

해결 방법은 구체적이고 실용적이어야 하며, 주어진 컨텍스트의 정보를 활용하세요."""
    
    else:  # information_search
        system_prompt = """당신은 정보 제공 전문가입니다. 주어진 컨텍스트를 바탕으로 사용자의 질의에 대한 정확하고 유용한 정보를 제공해주세요.

답변 형식:
1. 핵심 정보 요약
2. 상세 정보
3. 참고 자료

정보는 정확하고 관련성이 높아야 하며, 주어진 컨텍스트의 정보를 기반으로 해야 합니다."""
    
    # 사용자 프롬프트 구성
    user_prompt = f"""사용자 질의: {query}

참고 컨텍스트:
{context}

위 컨텍스트를 바탕으로 사용자 질의에 대한 답변을 생성해주세요. 
답변은 한국어로 작성하고, 참고 자료의 출처를 명시해주세요."""

    try:
        # LLM 호출
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        answer = response.content
        logger.info("LLM 기반 답변 생성 완료")
        
        return answer
        
    except Exception as e:
        logger.error(f"LLM 답변 생성 실패: {e}")
        # LLM 실패시 템플릿 기반 답변으로 대체
        return generate_template_answer(query, context_parts, sources, intent)


def generate_template_answer(query: str, context_parts: List[str], sources: List[str], intent: str) -> str:
    """템플릿 기반 답변 생성 (LLM 실패시 대체)"""
    
    if intent == "code_generation":
        return generate_code_answer(query, context_parts, sources)
    elif intent == "explanation_request":
        return generate_explanation_answer(query, context_parts, sources)
    elif intent == "troubleshooting":
        return generate_troubleshooting_answer(query, context_parts, sources)
    else:  # information_search
        return generate_information_answer(query, context_parts, sources)


def calculate_confidence_score(results: List[Dict[str, Any]], state: GraphState) -> float:
    """답변의 신뢰도 점수 계산"""
    
    if not results:
        return 0.0
    
    # 기본 신뢰도는 검색 결과의 평균 관련성 점수
    base_confidence = sum(r.get('relevance_score', 0.0) for r in results) / len(results)
    
    # 개인 학습 자료가 포함된 경우 신뢰도 증가
    personal_sources = [r for r in results if any(keyword in r.get('source', '').lower() 
                       for keyword in ["day", ".ipynb", "학습"])]
    if personal_sources:
        base_confidence += 0.1
    
    # 다양한 소스에서 일관된 정보가 있는 경우 신뢰도 증가
    if len(set(r.get('source', '') for r in results)) >= 2:
        base_confidence += 0.05
    
    # 신뢰도는 0.0 ~ 1.0 범위로 제한
    return min(1.0, max(0.0, base_confidence)) 