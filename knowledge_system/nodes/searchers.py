"""
검색 노드들

계층적 검색 전략에 따라 다양한 소스에서 정보를 검색하는 노드들입니다.
1. 그래프 검색 (개인 학습 자료)
2. 문서 검색 (공식 문서)  
3. 웹 검색 (최후 수단)
"""

import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from knowledge_system.models.state import GraphState, SearchResult
from knowledge_system.utils.relevance_evaluator import evaluate_single_result_relevance

logger = logging.getLogger(__name__)


def graph_searcher(state: GraphState) -> Dict[str, Any]:
    """
    그래프 DB 검색 노드 (개인 학습 자료 우선)
    
    Neo4j 그래프 DB에서 개인 학습 자료를 검색합니다.
    GraphCypherQAChain을 사용하여 자연어 질의를 Cypher 쿼리로 변환하고 실행합니다.
    LLM을 사용하여 관련성을 정확히 평가합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 정보 (graph_results, graph_relevance_score)
    """
    logger.info("그래프 DB 검색 시작")
    
    try:
        from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # analysis_result에서 직접 정보 추출
        entities = []
        query = state.query
        if state.analysis_result:
            entities = state.analysis_result.primary_entities
            
        logger.info(f"그래프 검색 질의: {query}")
        logger.info(f"추출된 엔티티: {entities}")
        
        # Neo4j 그래프 연결
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
        
        # LLM 초기화 (OpenAI 우선, 실패시 Google)
        llm = None
        try:
            llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
            logger.info("OpenAI LLM 초기화 성공")
        except Exception as e:
            logger.warning(f"OpenAI LLM 초기화 실패: {e}, Google LLM으로 대체")
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
                logger.info("Google LLM 초기화 성공")
            except Exception as e2:
                logger.error(f"Google LLM 초기화도 실패: {e2}")
                # LLM 초기화 실패시 빈 결과 반환
                return {
                    "graph_results": [],
                    "graph_relevance_score": 0.0,
                    "current_step": "graph_searched",
                    "error_message": "LLM 초기화 실패"
                }
        
        # 스키마 정보를 포함한 개선된 프롬프트 생성
        schema_info = """
        Neo4j Graph Database Schema:
        
        Node Types:
        1. Concept (core concepts)
           - properties: name(unique), description, source
           - examples: Graphs, Agent, State Management
        
        2. Component (implementations)
           - properties: name(unique), description, source  
           - examples: StateGraph, BM25Retriever, MemoryManager
        
        3. CodeExample (code examples)
           - properties: title(unique), description, code, source
           - examples: StateGraph usage, BM25 search example
        
        4. Tutorial (tutorials)
           - properties: title(unique), description, code, source
           - examples: GraphRAG tutorial, Multi-Agent implementation
        
        Relationship Types:
        1. IMPLEMENTS: (:Component)-[:IMPLEMENTS]->(:Concept)
           - example: StateGraph -IMPLEMENTS-> Graphs
        
        2. USES: (:CodeExample)-[:USES]->(:Component)
           - example: code_example -USES-> BM25Retriever
        
        3. APPLIES: (:Tutorial)-[:APPLIES]->(:Concept)
           - example: tutorial -APPLIES-> GraphRAG
        
        4. REQUIRES: (:Concept)-[:REQUIRES]->(:Concept)
           - example: Agents -REQUIRES-> Graphs
        
        5. INCLUDES: (:Tutorial)-[:INCLUDES]->(:Component)
           - example: tutorial -INCLUDES-> BM25Retriever
        
        Search Strategy:
        - Concept search: search in Concept nodes' name, description
        - Component search: search in Component nodes' name, description
        - Code example search: search in CodeExample nodes' title, description, code
        - Tutorial search: search in Tutorial nodes' title, description
        - Relationship-based search: find related nodes through relationships
        """
        
        # 개선된 GraphCypherQAChain 초기화
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            allow_dangerous_requests=True,
            verbose=False,
            return_intermediate_steps=True,
            top_k=5,  # 적절한 결과 수로 조정
        )
        
        # 영어로 Cypher 쿼리 생성을 명시적으로 지시하는 개선된 질의
        enhanced_query = f"""
        Based on the following Neo4j graph database schema, answer the user's question by generating an appropriate Cypher query in English:
        
        {schema_info}
        
        User Question: {query}
        
        IMPORTANT: 
        1. Generate Cypher queries in English only
        2. Use the schema information to create accurate queries
        3. Search for relevant concepts, components, code examples, and tutorials
        4. If the exact term is not found, search for related concepts using relationships
        5. Provide a comprehensive answer based on the graph data
        """
        
        logger.info(f"스키마 정보를 포함한 개선된 질의로 그래프 검색 실행")
        
        try:
            result_data = chain.invoke({"query": enhanced_query})
            result_text = result_data.get("result", "").strip()
            
            # 결과 검증
            if (result_text and 
                len(result_text) > 20 and 
                "정보가 없습니다" not in result_text and
                "알려드릴 수 있는 정보가 없습니다" not in result_text and
                "죄송합니다" not in result_text):
                
                logger.info(f"성공적인 응답 발견: {len(result_text)}자")
                
                # LLM을 사용하여 관련성 평가
                logger.info("LLM을 사용한 관련성 평가 시작")
                relevance_evaluation = evaluate_single_result_relevance(
                    query=query,
                    result_content=result_text,
                    llm=llm
                )
                
                # 검색 결과 생성
                search_result = SearchResult(
                    content=result_text,
                    source="Neo4j GraphDB",
                    relevance_score=relevance_evaluation.score,
                    metadata={
                        "query": query,
                        "original_query": query,
                        "entities": entities,
                        "type": "graph_cypher_qa",
                        "relevance_reasoning": relevance_evaluation.reasoning,
                        "intermediate_steps": result_data.get("intermediate_steps", [])
                    }
                )
                search_results = [search_result]
                
            else:
                logger.warning(f"의미없는 응답: {result_text[:50]}...")
                search_results = []
                
        except Exception as e:
            logger.error(f"그래프 검색 실행 중 오류: {e}")
            search_results = []
        
        # 관련성 점수 계산
        relevance_score = 0.0
        if search_results:
            relevance_score = search_results[0].relevance_score
            logger.info(f"그래프 검색 결과 관련성: {relevance_score:.2f}")
        else:
            logger.warning("그래프 검색 결과 없음")
        
        # SearchResult 객체들을 딕셔너리로 변환
        results_dict = [result.model_dump() for result in search_results]
        
        logger.info(f"그래프 검색 완료 - 결과 수: {len(search_results)}, 관련성: {relevance_score:.2f}")
        
        return {
            "graph_results": results_dict,
            "graph_relevance_score": relevance_score,
            "current_step": "graph_searched"
        }
        
    except ImportError as e:
        logger.error(f"필요한 라이브러리 import 실패: {e}")
        return {
            "graph_results": [],
            "graph_relevance_score": 0.0,
            "current_step": "graph_searched",
            "error_message": f"라이브러리 import 실패: {str(e)}"
        }
    except Exception as e:
        logger.error(f"그래프 검색 중 오류 발생: {str(e)}")
        return {
            "error_message": f"그래프 검색 실패: {str(e)}",
            "current_step": "error"
        }


def document_searcher(state: GraphState) -> Dict[str, Any]:
    """
    하이브리드 문서 검색 노드 (Vector Store + BM25)
    
    LangGraph 공식문서에서 Vector Store와 BM25를 결합한 하이브리드 검색을 수행합니다.
    LLM을 사용하여 검색 결과의 관련성을 정확히 평가합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 정보 (doc_results, doc_relevance_score)
    """
    logger.info("하이브리드 문서 검색 시작")
    
    try:
        import json
        import pickle
        from pathlib import Path
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document
        from langchain.retrievers import EnsembleRetriever
        
        query = state.query
        logger.info(f"검색 질의: {query}")
        
        # 1. Embeddings 초기화
        embeddings = None
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            logger.info("OpenAI 임베딩 사용")
        except Exception as e:
            logger.warning(f"OpenAI 임베딩 실패: {e}")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            logger.info("Google 임베딩으로 대체")
        
        # 2. Vector Store 로드
        vectorstore = Chroma(
            collection_name="langgraph_docs",
            embedding_function=embeddings,
            persist_directory="./data/chroma_db_langgraph"
        )
        logger.info("Vector Store 로드 완료")
        
        # 3. BM25 Retriever 로드
        bm25_retriever = None
        try:
            # BM25 pickle 로드 시도
            retriever_file = Path("data/bm25_index_langgraph/bm25_retriever.pkl")
            with open(retriever_file, 'rb') as f:
                bm25_retriever = pickle.load(f)
            logger.info("BM25 retriever pickle 로드 성공")
            
        except Exception as e:
            logger.warning(f"BM25 pickle 로드 실패: {e}, 재생성 중...")
            
            # 문서에서 BM25 재생성
            docs_file = Path("data/bm25_index_langgraph/documents.json")
            with open(docs_file, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
            
            # Document 객체로 변환
            documents = []
            for doc_data in docs_data:
                doc = Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data["metadata"]
                )
                documents.append(doc)
            
            # BM25 재생성
            bm25_retriever = BM25Retriever.from_documents(documents)
            logger.info("BM25 retriever 재생성 완료")
        
        # 4. 하이브리드 Retriever 생성
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(search_kwargs={"k": 4}),  # Vector Store
                bm25_retriever  # BM25
            ],
            weights=[0.6, 0.4]  # Vector Store 60%, BM25 40%
        )
        logger.info("하이브리드 retriever 생성 완료")
        
        # 5. 하이브리드 검색 수행
        try:
            search_results = ensemble_retriever.invoke(query)  # invoke 방식 사용
            logger.info(f"하이브리드 검색 완료: {len(search_results)}개 결과")
        except Exception as e:
            # invoke 실패 시 get_relevant_documents로 대체
            logger.warning(f"invoke 실패: {e}, get_relevant_documents 사용")
            search_results = ensemble_retriever.get_relevant_documents(query)
        
        if not search_results:
            logger.warning("검색 결과 없음")
            return {
                "doc_results": [],
                "doc_relevance_score": 0.0,
                "current_step": "document_searched"
            }
        
        # 6. LLM 초기화 (관련성 평가용)
        llm = None
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4.1")
            logger.info("OpenAI LLM 사용")
        except Exception as e:
            logger.warning(f"OpenAI LLM 실패: {e}")
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            logger.info("Google LLM으로 대체")
        
        # 7. 검색 결과 처리 및 LLM 평가 (상위 5개만)
        processed_results = []
        total_relevance = 0.0
        
        for i, doc in enumerate(search_results[:5]):  # 상위 5개만 처리
            logger.info(f"결과 {i+1} 처리 중...")
            
            # LLM을 사용한 관련성 평가
            try:
                from knowledge_system.utils.relevance_evaluator import evaluate_single_result_relevance
                relevance_result = evaluate_single_result_relevance(
                    query=query,
                    result_content=doc.page_content,
                    llm=llm
                )
                
                relevance_score = relevance_result.score
                reasoning = relevance_result.reasoning
                
                logger.info(f"LLM 평가 점수: {relevance_score:.2f}")
                
            except Exception as e:
                logger.error(f"LLM 평가 실패: {e}")
                # 기본 점수 사용 (fallback)
                relevance_score = 0.6
                reasoning = f"LLM 평가 실패로 인한 기본 점수"
            
            # 결과 구성
            result = {
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "relevance_score": float(relevance_score),
                    "relevance_reasoning": reasoning,
                    "search_rank": i + 1,
                    "source_type": "langgraph_hybrid_search",
                    "search_method": "vector_store_bm25_ensemble"
                }
            }
            
            processed_results.append(result)
            total_relevance += relevance_score
        
        # 8. 평균 관련성 점수 계산
        avg_relevance_score = total_relevance / len(processed_results) if processed_results else 0.0
        
        logger.info(f"하이브리드 검색 완료: {len(processed_results)}개 결과, 평균 관련성: {avg_relevance_score:.3f}")
        
        return {
            "doc_results": processed_results,
            "doc_relevance_score": avg_relevance_score,
            "current_step": "document_searched"
        }
        
    except ImportError as e:
        logger.error(f"필요한 라이브러리 누락: {e}")
        return {
            "doc_results": [],
            "doc_relevance_score": 0.0,
            "current_step": "document_searched"
        }
        
    except Exception as e:
        logger.error(f"하이브리드 검색 오류: {e}")
        return {
            "doc_results": [],
            "doc_relevance_score": 0.0,
            "current_step": "document_searched"
        }


def web_searcher(state: GraphState) -> Dict[str, Any]:
    """
    웹 검색 노드 (Tavily Search 사용)
    
    그래프 검색과 문서 검색으로 충분한 결과를 얻지 못한 경우
    Tavily Search API를 통해 최신 웹 정보를 수집합니다.
    LLM을 사용하여 검색 결과의 관련성을 정확히 평가합니다.
    
    Args:
        state: 현재 그래프 상태
        
    Returns:
        업데이트된 상태 정보 (web_results, web_relevance_score)
    """
    logger.info("Tavily 웹 검색 시작")
    
    try:
        from langchain_tavily import TavilySearch
        from langchain_openai import ChatOpenAI
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        query = state.query
        logger.info(f"웹 검색 질의: {query}")
        
        # Tavily Search 초기화
        search_web = TavilySearch(
            max_results=5,
            topic="general",
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
            include_images=False
        )
        
        # 웹 검색 수행
        search_results = search_web.invoke(query)
        logger.info(f"Tavily 검색 완료: {search_results.get('total', 0)}개 결과 중 {len(search_results.get('results', []))}개 반환")
        
        if not search_results.get('results'):
            logger.warning("웹 검색 결과 없음")
            return {
                "web_results": [],
                "web_relevance_score": 0.0,
                "current_step": "web_searched"
            }
        
        # 검색 결과 처리
        processed_results = []
        
        for i, result in enumerate(search_results['results']):
            logger.info(f"웹 결과 {i+1} 처리 중...")
            
            # 검색 결과 내용 구성
            content = result.get('content', '')
            title = result.get('title', '')
            url = result.get('url', '')
            score = result.get('score', 0.0)
            
            # 제목과 내용을 합쳐서 전체 내용 구성
            full_content = f"제목: {title}\n내용: {content}"
            
            # Tavily 점수를 기반으로 관련성 점수 계산
            relevance_score = min(0.8, max(0.3, float(score)))
            
            # 결과 구성
            web_result = {
                "content": full_content,
                "metadata": {
                    "title": title,
                    "url": url,
                    "tavily_score": float(score),
                    "relevance_score": float(relevance_score),
                    "search_rank": i + 1,
                    "source_type": "web_search",
                    "search_method": "tavily_api",
                    "pub_date": result.get('pubDate', ''),
                    "domain": url.split('/')[2] if '/' in url else url
                }
            }
            
            processed_results.append(web_result)
        
        # 평균 관련성 점수 계산
        avg_relevance_score = sum(result["metadata"]["relevance_score"] for result in processed_results) / len(processed_results) if processed_results else 0.0
        
        logger.info(f"웹 검색 완료: {len(processed_results)}개 결과, 평균 관련성: {avg_relevance_score:.3f}")
        
        return {
            "web_results": processed_results,
            "web_relevance_score": avg_relevance_score,
            "current_step": "web_searched"
        }
        
    except ImportError as e:
        logger.error(f"Tavily 라이브러리 누락: {e}")
        return {
            "web_results": [],
            "web_relevance_score": 0.0,
            "current_step": "web_searched",
            "error_message": f"Tavily 라이브러리 누락: {str(e)}"
        }
        
    except Exception as e:
        logger.error(f"웹 검색 오류: {e}")
        return {
            "web_results": [],
            "web_relevance_score": 0.0,
            "current_step": "web_searched",
            "error_message": f"웹 검색 실패: {str(e)}"
        } 