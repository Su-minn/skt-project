"""
LangGraph 기반 개인화 지식 관리 시스템

계층적 검색 전략을 통해 개인 학습 자료 → 공식 문서 → 웹 검색 순으로
최적의 답변을 제공하는 AI 어시스턴트입니다.
"""

__version__ = "0.1.0"
__author__ = "Knowledge System Team"

from .models.state import GraphState
from .graph.builder import create_knowledge_graph

__all__ = [
    "GraphState",
    "create_knowledge_graph",
] 