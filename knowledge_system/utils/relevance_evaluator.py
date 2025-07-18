"""
관련성 평가 모듈

LLM을 사용하여 사용자 질의와 검색 결과 간의 관련성을 정확히 평가합니다.
Structured Output과 Pydantic 스키마를 사용하여 일관된 숫자 결과를 보장합니다.
"""

import logging
from typing import List, Optional
from knowledge_system.models.state import RelevanceScore

logger = logging.getLogger(__name__)


def evaluate_relevance(
    query: str,
    search_results: List[str],
    llm,
    max_retries: int = 3
) -> float:
    """
    LLM을 사용하여 질의와 검색 결과들의 관련성을 평가합니다.
    
    Args:
        query: 사용자 질의
        search_results: 검색 결과 텍스트 리스트
        llm: 초기화된 LLM 객체 (ChatOpenAI 또는 ChatGoogleGenerativeAI)
        max_retries: 최대 재시도 횟수
        
    Returns:
        0.0-1.0 사이의 관련성 점수
    """
    if not search_results:
        return 0.0
    
    try:
        # 모든 검색 결과를 하나의 텍스트로 결합
        combined_results = "\n\n---\n\n".join(search_results)
        
        # 프롬프트 생성
        prompt_template = """당신은 정보 검색 품질을 평가하는 전문가입니다.

사용자 질의와 검색 결과를 비교하여 관련성을 정확히 평가해주세요.

**사용자 질의:**
{query}

**검색 결과:**
{results}

**평가 기준:**
- 1.0: 질의에 대한 완벽하고 직접적인 답변
- 0.8-0.9: 질의와 매우 관련성이 높고 유용한 정보
- 0.6-0.7: 질의와 관련이 있지만 부분적으로 유용
- 0.4-0.5: 질의와 약간의 관련성은 있지만 제한적
- 0.2-0.3: 질의와 거의 관련성이 없음
- 0.0-0.1: 질의와 전혀 관련성이 없음

검색 결과가 사용자 질의에 얼마나 적절히 답변하는지 숫자로 평가하고, 그 이유를 설명해주세요."""

        prompt = prompt_template.format(
            query=query,
            results=combined_results[:2000]  # 토큰 제한을 위해 길이 제한
        )
        
        # LLM에 structured output 요청
        for attempt in range(max_retries):
            try:
                # withStructuredOutput 또는 직접 구조화된 출력 요청
                if hasattr(llm, 'with_structured_output'):
                    # LangChain의 새로운 structured output 방식
                    structured_llm = llm.with_structured_output(RelevanceScore)
                    result = structured_llm.invoke(prompt)
                    
                    if isinstance(result, RelevanceScore):
                        logger.info(f"관련성 평가 완료: {result.score:.2f} - {result.reasoning[:50]}...")
                        return result.score
                        
                else:
                    # 수동으로 구조화된 응답 요청
                    manual_prompt = f"""{prompt}

반드시 다음 JSON 형식으로만 응답해주세요:
{{
    "score": 0.0,
    "reasoning": "평가 이유"
}}"""
                    
                    response = llm.invoke(manual_prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    # JSON 파싱 시도
                    import json
                    import re
                    
                    # JSON 부분 추출
                    json_match = re.search(r'\{[^}]*\}', response_text)
                    if json_match:
                        json_str = json_match.group()
                        parsed = json.loads(json_str)
                        
                        score = float(parsed.get('score', 0.0))
                        reasoning = parsed.get('reasoning', 'No reasoning provided')
                        
                        # 점수 범위 검증
                        score = max(0.0, min(1.0, score))
                        
                        logger.info(f"관련성 평가 완료: {score:.2f} - {reasoning[:50]}...")
                        return score
                        
            except Exception as e:
                logger.warning(f"관련성 평가 시도 {attempt + 1} 실패: {e}")
                if attempt == max_retries - 1:
                    logger.error("모든 관련성 평가 시도 실패, 기본값 0.5 반환")
                    return 0.5
                    
        return 0.5  # 기본값
        
    except Exception as e:
        logger.error(f"관련성 평가 중 오류: {e}")
        return 0.5  # 기본값


def evaluate_single_result_relevance(
    query: str,
    result_content: str,
    llm,
    max_retries: int = 3
) -> RelevanceScore:
    """
    단일 검색 결과의 관련성을 상세히 평가합니다.
    
    Args:
        query: 사용자 질의
        result_content: 검색 결과 내용
        llm: 초기화된 LLM 객체
        max_retries: 최대 재시도 횟수
        
    Returns:
        RelevanceScore 객체 (점수와 이유 포함)
    """
    try:
        prompt_template = """다음 사용자 질의와 검색 결과의 관련성을 정확히 평가해주세요.

**사용자 질의:**
{query}

**검색 결과:**
{content}

**평가 기준:**
- 1.0: 질의에 대한 완벽하고 직접적인 답변
- 0.8-0.9: 질의와 매우 관련성이 높고 유용한 정보  
- 0.6-0.7: 질의와 관련이 있지만 부분적으로 유용
- 0.4-0.5: 질의와 약간의 관련성은 있지만 제한적
- 0.2-0.3: 질의와 거의 관련성이 없음
- 0.0-0.1: 질의와 전혀 관련성이 없음

0.0에서 1.0 사이의 정확한 점수와 평가 이유를 제공해주세요."""

        prompt = prompt_template.format(
            query=query,
            content=result_content[:1500]  # 토큰 제한
        )
        
        for attempt in range(max_retries):
            try:
                if hasattr(llm, 'with_structured_output'):
                    structured_llm = llm.with_structured_output(RelevanceScore)
                    result = structured_llm.invoke(prompt)
                    
                    if isinstance(result, RelevanceScore):
                        return result
                else:
                    # 수동 JSON 파싱
                    manual_prompt = f"""{prompt}

반드시 다음 JSON 형식으로만 응답해주세요:
{{
    "score": 0.0,
    "reasoning": "점수 산정 이유를 상세히 설명"
}}"""
                    
                    response = llm.invoke(manual_prompt)
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    
                    import json
                    import re
                    
                    json_match = re.search(r'\{[^}]*\}', response_text)
                    if json_match:
                        json_str = json_match.group()
                        parsed = json.loads(json_str)
                        
                        score = max(0.0, min(1.0, float(parsed.get('score', 0.5))))
                        reasoning = parsed.get('reasoning', 'No reasoning provided')
                        
                        return RelevanceScore(score=score, reasoning=reasoning)
                        
            except Exception as e:
                logger.warning(f"단일 결과 관련성 평가 시도 {attempt + 1} 실패: {e}")
                
        # 모든 시도 실패시 기본값
        return RelevanceScore(
            score=0.5,
            reasoning="관련성 평가 실패로 인한 기본 점수"
        )
        
    except Exception as e:
        logger.error(f"단일 결과 관련성 평가 중 오류: {e}")
        return RelevanceScore(
            score=0.5,
            reasoning=f"오류 발생: {str(e)}"
        ) 