"""
LangGraph 지식 관리 시스템 메인 실행 파일

사용자와 상호작용하며 질의응답을 처리하는 CLI 인터페이스를 제공합니다.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
import sys
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from knowledge_system.models.state import GraphState
from knowledge_system.graph.builder import (
    create_knowledge_graph, 
    create_simple_graph,
    visualize_graph,
    get_graph_info
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('knowledge_system.log')
    ]
)

logger = logging.getLogger(__name__)


class KnowledgeSystemCLI:
    """지식 관리 시스템 CLI 인터페이스"""
    
    def __init__(self, use_simple_graph: bool = False):
        """
        CLI 인터페이스 초기화
        
        Args:
            use_simple_graph: 간단한 그래프 사용 여부
        """
        self.use_simple_graph = use_simple_graph
        self.graph = None
        self.session_history = []
        
    def initialize_system(self) -> bool:
        """
        시스템 초기화 및 그래프 생성
        
        Returns:
            초기화 성공 여부
        """
        try:
            logger.info("지식 관리 시스템 초기화 시작")
            
            # 그래프 생성
            if self.use_simple_graph:
                logger.info("간단한 테스트 그래프 생성")
                self.graph = create_simple_graph()
            else:
                logger.info("전체 기능 그래프 생성")
                self.graph = create_knowledge_graph()
            
            # 그래프 정보 출력
            graph_info = get_graph_info(self.graph)
            logger.info(f"그래프 정보: {graph_info}")
            
            # 그래프 시각화
            visualize_graph(self.graph)
            
            logger.info("시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {str(e)}")
            return False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        사용자 질의를 처리합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            처리 결과
        """
        try:
            logger.info(f"질의 처리 시작: {query}")
            
            # 초기 상태 생성
            initial_state = GraphState(query=query)
            
            # 그래프 실행
            final_state = self.graph.invoke(initial_state)
            
            # LangGraph는 GraphState 객체를 반환하므로 객체 속성에서 값 추출
            try:
                final_answer = getattr(final_state, 'answer', None) or "답변을 생성할 수 없습니다."
                sources = getattr(final_state, 'sources', None) or []
                confidence_score = getattr(final_state, 'confidence_score', None) or 0.0
                entities = getattr(final_state.analysis_result, 'primary_entities', None) if final_state.analysis_result else []
                intent = getattr(final_state.analysis_result, 'intent_category', None) if final_state.analysis_result else "정보검색"
            except AttributeError as attr_error:
                logger.warning(f"상태 속성 접근 오류: {attr_error}, 기본값 사용")
                final_answer = "답변을 생성할 수 없습니다."
                sources = []
                confidence_score = 0.0
                entities = []
                intent = "정보검색"
            
            # 세션 히스토리에 추가
            self.session_history.append({
                "query": query,
                "response": final_answer,
                "sources": sources,
                "confidence": confidence_score
            })
            
            logger.info("질의 처리 완료")
            
            return {
                "success": True,
                "query": query,
                "answer": final_answer,
                "sources": sources or [],
                "confidence_score": confidence_score or 0.0,
                "entities": entities or [],
                "intent": intent or "정보검색"
            }
            
        except Exception as e:
            logger.error(f"질의 처리 중 오류 발생: {str(e)}")
            
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "answer": f"죄송합니다. 질의 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    def run_interactive_mode(self):
        """대화형 모드 실행"""
        
        print("\n" + "="*60)
        print("🤖 LangGraph 개인화 지식 관리 시스템")
        print("="*60)
        print("📚 계층적 검색: 개인자료 → 공식문서 → 웹검색")
        print("💡 질문하시면 최적의 답변을 제공해드립니다!")
        print("❌ 종료하려면 'quit', 'exit', '종료'를 입력하세요.")
        print("="*60 + "\n")
        
        while True:
            try:
                # 사용자 입력 받기
                query = input("\n🔍 질문을 입력하세요: ").strip()
                
                # 종료 명령 확인
                if query.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 지식 관리 시스템을 종료합니다. 좋은 하루 되세요!")
                    break
                
                # 빈 입력 처리
                if not query:
                    print("❓ 질문을 입력해주세요.")
                    continue
                
                # 질의 처리
                print("\n🔄 질의를 처리 중입니다...")
                result = self.process_query(query)
                
                # 결과 출력
                self._display_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
    
    def _display_result(self, result: Dict[str, Any]):
        """결과를 포맷팅하여 출력합니다."""
        
        print("\n" + "="*60)
        
        if result["success"]:
            print("✅ 답변:")
            print("-" * 40)
            print(result["answer"])
            
            if result["sources"]:
                print(f"\n📖 참고 자료 ({len(result['sources'])}개):")
                for i, source in enumerate(result["sources"], 1):
                    print(f"  {i}. {source}")
            
            print(f"\n📊 정보:")
            print(f"  • 신뢰도: {result['confidence_score']:.2f}/1.0")
            print(f"  • 의도: {result['intent']}")
            if result["entities"]:
                print(f"  • 핵심 키워드: {', '.join(result['entities'])}")
                
        else:
            print("❌ 오류:")
            print("-" * 40)
            print(result["answer"])
            
        print("="*60)
    
    def run_batch_test(self, test_queries: list):
        """배치 테스트 실행"""
        
        print("\n🧪 배치 테스트 시작")
        print("="*60)
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] 테스트: {query}")
            print("-" * 40)
            
            result = self.process_query(query)
            results.append(result)
            
            if result["success"]:
                print(f"✅ 성공 (신뢰도: {result['confidence_score']:.2f})")
                print(f"📝 답변: {result['answer'][:100]}...")
            else:
                print(f"❌ 실패: {result['error']}")
        
        # 결과 요약
        success_count = sum(1 for r in results if r["success"])
        avg_confidence = sum(r.get("confidence_score", 0) for r in results if r["success"]) / max(success_count, 1)
        
        print(f"\n📊 테스트 결과 요약:")
        print(f"  • 성공률: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
        print(f"  • 평균 신뢰도: {avg_confidence:.2f}")
        print("="*60)
        
        return results


def run_simple_test():
    """간단한 테스트 실행"""
    
    print("🚀 간단한 테스트 시작")
    
    # 시스템 초기화
    cli = KnowledgeSystemCLI(use_simple_graph=True)
    
    if not cli.initialize_system():
        print("❌ 시스템 초기화 실패")
        return
    
    # 테스트 질의들
    test_queries = [
        "LangGraph란 무엇인가요?",
        "StateGraph 사용법을 알려주세요",
        "ReAct Agent 구현 방법은?",
        "노드와 엣지를 어떻게 연결하나요?"
    ]
    
    # 배치 테스트 실행
    cli.run_batch_test(test_queries)


def main():
    """메인 함수"""
    
    # 시스템 초기화
    cli = KnowledgeSystemCLI(use_simple_graph=False)
    
    if not cli.initialize_system():
        print("❌ 시스템 초기화에 실패했습니다.")
        return
    
    # CLI 모드 선택
    print("\n🎯 실행 모드를 선택하세요:")
    print("1. 대화형 모드 (기본)")
    print("2. 간단한 테스트")
    print("3. 시스템 정보만 출력")
    
    try:
        choice = input("\n선택 (1-3, 기본값: 1): ").strip()
        
        if choice == "2":
            run_simple_test()
        elif choice == "3":
            graph_info = get_graph_info(cli.graph)
            print(f"\n📋 시스템 정보:")
            for key, value in graph_info.items():
                print(f"  • {key}: {value}")
        else:
            cli.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 시스템을 종료합니다.")
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main() 