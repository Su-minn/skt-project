#!/usr/bin/env python3
"""
Neo4j 기반 LangGraph RAG 시스템

기능:
- 임베딩 기반 유사도 검색
- 그래프 관계를 활용한 확장 검색 
- 컨텍스트 정보 풍부화
- 다중 검색 전략 지원

사용법:
    python scripts/neo4j_rag_system.py
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    concept_name: str
    node_type: str
    score: float
    metadata: Dict[str, Any]
    related_concepts: List[str] = None

class Neo4jLangGraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        # Neo4j Vector Store 초기화
        self.setup_vector_stores()
        
        # RAG 체인 설정
        self.setup_rag_chain()
    
    def setup_vector_stores(self):
        """벡터 스토어 설정"""
        print("🔧 벡터 스토어 설정 중...")
        
        # 개념 노드용 벡터 스토어
        self.concept_vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="concept_embedding_index",
            node_label="Concept",
            text_node_property="definition",
            embedding_node_property="embedding",
            retrieval_query="""
            RETURN node.definition + ' ' + apoc.text.join(node.characteristics, ' ') + ' ' + node.usage AS text,
                   score,
                   {
                       name: node.name,
                       node_type: node.node_type,
                       characteristics: node.characteristics,
                       usage: node.usage,
                       source_file: node.source_file,
                       chunk_id: node.chunk_id
                   } AS metadata
            """
        )
        
        # 청크용 벡터 스토어 (하위 섹션들)
        self.chunk_vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="chunk_embedding_index",
            node_label="Chunk",
            text_node_property="content",
            embedding_node_property="embedding",
            retrieval_query="""
            RETURN node.content AS text,
                   score,
                   {
                       title: node.title,
                       parent_concept: node.parent_concept,
                       chunk_type: node.chunk_type,
                       source_file: node.source_file,
                       chunk_id: node.chunk_id
                   } AS metadata
            """
        )
        
        print("✅ 벡터 스토어 설정 완료")
    
    def setup_rag_chain(self):
        """RAG 체인 설정"""
        template = """
당신은 LangGraph 전문가입니다. 제공된 지식을 바탕으로 질문에 정확하고 구체적으로 답변해주세요.

**지식 베이스:**
{context}

**질문:** {question}

**답변 지침:**
1. 제공된 정보를 바탕으로 정확하게 답변하세요
2. 관련된 개념들과의 관계를 설명하세요  
3. 가능하면 구체적인 예시나 코드를 포함하세요
4. 추가 학습이 필요한 선행 개념이 있다면 언급하세요

**답변:**
"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def enhanced_search(self, query: str, k: int = 5) -> List[SearchResult]:
        """확장 검색: 임베딩 + 그래프 관계 활용"""
        print(f"🔍 확장 검색 시작: '{query}'")
        
        # 1. 임베딩 기반 초기 검색
        concept_results = self.concept_vector_store.similarity_search_with_score(query, k=k)
        chunk_results = self.chunk_vector_store.similarity_search_with_score(query, k=k//2)
        
        # 2. 검색 결과 통합
        all_results = []
        
        # 개념 결과 처리
        for doc, score in concept_results:
            result = SearchResult(
                content=f"**{doc.metadata['name']}**\n정의: {doc.page_content}\n특징: {', '.join(doc.metadata.get('characteristics', []))}\n활용: {doc.metadata.get('usage', '')}",
                concept_name=doc.metadata['name'],
                node_type=doc.metadata.get('node_type', 'concept'),
                score=score,
                metadata=doc.metadata
            )
            all_results.append(result)
        
        # 청크 결과 처리
        for doc, score in chunk_results:
            result = SearchResult(
                content=f"**{doc.metadata['title']}** ({doc.metadata['parent_concept']})\n{doc.page_content}",
                concept_name=doc.metadata['parent_concept'],
                node_type="chunk",
                score=score,
                metadata=doc.metadata
            )
            all_results.append(result)
        
        # 3. 관련 개념 확장 검색
        expanded_results = self.expand_with_relationships(all_results)
        
        print(f"   📊 초기 결과: {len(all_results)}개, 확장 후: {len(expanded_results)}개")
        
        return expanded_results[:k*2]  # 확장된 결과에서 상위 선택
    
    def expand_with_relationships(self, initial_results: List[SearchResult]) -> List[SearchResult]:
        """관계를 활용한 검색 결과 확장"""
        expanded_results = initial_results.copy()
        
        for result in initial_results:
            if result.node_type == "chunk":
                continue  # 청크는 확장하지 않음
                
            concept_name = result.concept_name
            
            # 관련 개념들 검색
            related_query = """
            MATCH (c {name: $concept_name})
            MATCH (c)-[r]-(related)
            WHERE related:Concept OR related:Tutorial OR related:Pattern OR related:Tool
            RETURN related.name AS name, 
                   related.definition AS definition,
                   related.node_type AS node_type,
                   type(r) AS relationship_type,
                   related.characteristics AS characteristics,
                   related.usage AS usage
            LIMIT 3
            """
            
            try:
                related_concepts = self.graph.query(
                    related_query, 
                    params={"concept_name": concept_name}
                )
                
                result.related_concepts = []
                
                for related in related_concepts:
                    if related['name'] != concept_name:  # 자기 자신 제외
                        related_result = SearchResult(
                            content=f"**{related['name']}** (관련: {related['relationship_type']})\n정의: {related['definition']}\n활용: {related.get('usage', '')}",
                            concept_name=related['name'],
                            node_type=related['node_type'],
                            score=result.score * 0.8,  # 관련 개념은 점수 감소
                            metadata={
                                'name': related['name'],
                                'node_type': related['node_type'],
                                'relationship_type': related['relationship_type'],
                                'characteristics': related.get('characteristics', []),
                                'usage': related.get('usage', '')
                            }
                        )
                        expanded_results.append(related_result)
                        result.related_concepts.append(related['name'])
                        
            except Exception as e:
                print(f"⚠️ 관계 확장 중 에러 ({concept_name}): {e}")
        
        return expanded_results
    
    def get_learning_path(self, concept_name: str) -> Dict[str, List[str]]:
        """학습 경로 추천"""
        path_query = """
        MATCH (target {name: $concept_name})
        
        // 선행 개념들 (배워야 할 것들)
        OPTIONAL MATCH (prereq)-[:PREREQUISITE]->(target)
        
        // 후속 개념들 (다음에 배울 것들)  
        OPTIONAL MATCH (target)-[:PREREQUISITE]->(next)
        
        // 관련 개념들
        OPTIONAL MATCH (target)-[:RELATES_TO]-(related)
        
        RETURN 
            collect(DISTINCT prereq.name) AS prerequisites,
            collect(DISTINCT next.name) AS next_concepts,
            collect(DISTINCT related.name) AS related_concepts
        """
        
        try:
            result = self.graph.query(path_query, params={"concept_name": concept_name})
            if result:
                return {
                    "prerequisites": [name for name in result[0]['prerequisites'] if name],
                    "next_concepts": [name for name in result[0]['next_concepts'] if name], 
                    "related_concepts": [name for name in result[0]['related_concepts'] if name]
                }
        except Exception as e:
            print(f"⚠️ 학습 경로 조회 에러: {e}")
        
        return {"prerequisites": [], "next_concepts": [], "related_concepts": []}
    
    def format_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과를 컨텍스트로 포맷팅"""
        context_parts = []
        
        # 점수 기준으로 정렬
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            context_part = f"[정보 {i}] {result.content}"
            
            # 관련 개념 정보 추가
            if result.related_concepts:
                context_part += f"\n관련 개념: {', '.join(result.related_concepts)}"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def ask(self, question: str, search_strategy: str = "enhanced") -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        print(f"❓ 질문: {question}")
        
        # 검색 전략 선택
        if search_strategy == "enhanced":
            search_results = self.enhanced_search(question)
        elif search_strategy == "concept_only":
            concept_results = self.concept_vector_store.similarity_search_with_score(question, k=5)
            search_results = [
                SearchResult(
                    content=doc.page_content,
                    concept_name=doc.metadata['name'],
                    node_type=doc.metadata.get('node_type', 'concept'),
                    score=score,
                    metadata=doc.metadata
                ) for doc, score in concept_results
            ]
        else:
            search_results = self.enhanced_search(question)
        
        # 컨텍스트 생성
        context = self.format_context(search_results)
        
        # LLM으로 답변 생성
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm  
            | StrOutputParser()
        )
        
        answer = chain.invoke(question)
        
        # 주요 개념의 학습 경로 추가
        learning_paths = {}
        for result in search_results[:3]:  # 상위 3개 개념만
            if result.node_type in ["concept", "tutorial", "pattern"]:
                learning_paths[result.concept_name] = self.get_learning_path(result.concept_name)
        
        return {
            "answer": answer,
            "search_results": search_results,
            "learning_paths": learning_paths,
            "context_used": context
        }
    
    def get_concept_overview(self, concept_name: str) -> Dict[str, Any]:
        """특정 개념의 상세 정보"""
        overview_query = """
        MATCH (c {name: $concept_name})
        OPTIONAL MATCH (c)-[:HAS_SUBSECTION]->(sub:Chunk)
        RETURN c.name AS name,
               c.definition AS definition, 
               c.characteristics AS characteristics,
               c.usage AS usage,
               c.code_examples AS code_examples,
               c.node_type AS node_type,
               collect(sub.title) AS subsections
        """
        
        try:
            result = self.graph.query(overview_query, params={"concept_name": concept_name})
            if result:
                concept_data = result[0]
                learning_path = self.get_learning_path(concept_name)
                
                return {
                    "concept_info": concept_data,
                    "learning_path": learning_path
                }
        except Exception as e:
            print(f"⚠️ 개념 조회 에러: {e}")
        
        return None
    
    def interactive_chat(self):
        """대화형 인터페이스"""
        print("🤖 LangGraph RAG 시스템에 오신 것을 환영합니다!")
        print("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ 질문: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료']:
                    print("👋 감사합니다!")
                    break
                
                if not question:
                    continue
                
                # 특별 명령어 처리
                if question.startswith("/개념 "):
                    concept_name = question[3:].strip()
                    overview = self.get_concept_overview(concept_name)
                    if overview:
                        info = overview["concept_info"]
                        print(f"\n📖 **{info['name']}** ({info['node_type']})")
                        print(f"정의: {info['definition']}")
                        if info['characteristics']:
                            print(f"특징: {', '.join(info['characteristics'])}")
                        print(f"활용: {info['usage']}")
                        
                        # 학습 경로
                        path = overview["learning_path"]
                        if path['prerequisites']:
                            print(f"선행 개념: {', '.join(path['prerequisites'])}")
                        if path['next_concepts']:
                            print(f"후속 개념: {', '.join(path['next_concepts'])}")
                    else:
                        print(f"❌ '{concept_name}' 개념을 찾을 수 없습니다.")
                    continue
                
                # 일반 질문 처리
                result = self.ask(question)
                
                print(f"\n💡 **답변:**")
                print(result["answer"])
                
                print(f"\n📊 **참고된 개념들:**")
                for i, search_result in enumerate(result["search_results"][:3], 1):
                    print(f"{i}. {search_result.concept_name} (점수: {search_result.score:.3f})")
                
                # 학습 경로 제안
                if result["learning_paths"]:
                    print(f"\n🎯 **추천 학습 경로:**")
                    for concept, path in result["learning_paths"].items():
                        if path['prerequisites']:
                            print(f"• {concept} 학습 전 필요: {', '.join(path['prerequisites'])}")
                
            except KeyboardInterrupt:
                print("\n👋 감사합니다!")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    # Neo4j 연결 정보 확인
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("❌ Neo4j 연결 정보가 필요합니다 (.env 파일 확인)")
        return
    
    try:
        # RAG 시스템 초기화
        print("🚀 Neo4j LangGraph RAG 시스템 초기화 중...")
        rag_system = Neo4jLangGraphRAG(neo4j_uri, neo4j_username, neo4j_password)
        
        # 대화형 모드 시작
        rag_system.interactive_chat()
        
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")

if __name__ == "__main__":
    main() 