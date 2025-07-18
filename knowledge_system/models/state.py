"""
그래프 상태 모델 정의

LangGraph에서 사용할 상태와 검색 결과 모델을 정의합니다.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryAnalysisResult(BaseModel):
    """질의 분석 결과"""
    primary_entities: List[str] = Field(description="주요 엔티티 목록")
    intent_category: str = Field(description="질의 의도 카테고리")
    context_hint: str = Field(description="추가 컨텍스트 힌트")


class RelevanceScore(BaseModel):
    """관련성 점수 평가 결과"""
    score: float = Field(
        description="0.0에서 1.0 사이의 관련성 점수",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="점수 산정 이유")


class SearchResult(BaseModel):
    """검색 결과 항목"""
    content: str = Field(description="검색된 내용")
    source: str = Field(description="검색 소스")
    relevance_score: float = Field(description="관련성 점수", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")


class GraphState(BaseModel):
    """그래프 상태 모델"""
    # 입력 질의
    query: str = Field(description="사용자 질의")
    
    # 분석 결과
    analysis_result: Optional[QueryAnalysisResult] = Field(default=None, description="질의 분석 결과")
    
    # 검색 결과들
    graph_results: List[Dict[str, Any]] = Field(default_factory=list, description="그래프 검색 결과")
    doc_results: List[Dict[str, Any]] = Field(default_factory=list, description="문서 검색 결과")
    web_results: List[Dict[str, Any]] = Field(default_factory=list, description="웹 검색 결과")
    
    # 관련성 점수들
    graph_relevance_score: float = Field(default=0.0, description="그래프 검색 관련성 점수")
    doc_relevance_score: float = Field(default=0.0, description="문서 검색 관련성 점수")
    web_relevance_score: float = Field(default=0.0, description="웹 검색 관련성 점수")
    
    # 최종 답변
    answer: str = Field(default="", description="생성된 답변")
    confidence_score: float = Field(default=0.0, description="답변 신뢰도")
    sources: List[str] = Field(default_factory=list, description="답변에 사용된 소스")
    
    # 워크플로우 상태
    current_step: str = Field(default="start", description="현재 워크플로우 단계")
    error_message: str = Field(default="", description="오류 메시지")

    def get_best_results(self, max_results: int = 3) -> List[Dict[str, Any]]:
        """최고 점수의 검색 결과들을 반환"""
        all_results = []
        all_results.extend(self.graph_results)
        all_results.extend(self.doc_results)
        all_results.extend(self.web_results)
        
        # 관련성 점수로 정렬
        sorted_results = sorted(all_results, key=lambda x: x.get('relevance_score', 0), reverse=True)
        return sorted_results[:max_results]
    
    def has_sufficient_results(self, min_score: float = 0.7) -> bool:
        """충분한 품질의 검색 결과가 있는지 확인"""
        best_results = self.get_best_results(1)
        return len(best_results) > 0 and best_results[0].get('relevance_score', 0) >= min_score
    
    def calculate_confidence_score(self) -> float:
        """전체 신뢰도 점수 계산"""
        scores = [self.graph_relevance_score, self.doc_relevance_score, self.web_relevance_score]
        valid_scores = [s for s in scores if s > 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0 