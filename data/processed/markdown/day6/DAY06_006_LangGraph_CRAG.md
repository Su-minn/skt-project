#   LangGraph 활용 - Corrective RAG (CRAG)

---

## 1. 환경 설정

`(1) Env 환경변수`


```python
from dotenv import load_dotenv
load_dotenv()
```




    True



`(2) 기본 라이브러리`


```python
import re
import os, json
from pprint import pprint
import time
import uuid

import warnings
warnings.filterwarnings("ignore")
```

`(3) Langsmith tracing 설정`


```python
# Langsmith tracing 여부를 확인 (true: langsmith 추적 활성화, false: langsmith 추적 비활성화)
import os
print(os.getenv('LANGSMITH_TRACING'))
```

    true


## 2. Corrective RAG (CRAG) 구현
- CRAG (Corrective Retrieval-Augmented Generation) 
- 논문: https://arxiv.org/pdf/2401.15884

- 주요 과정: 검색 -> 평가 -> 지식 정제 또는 웹 검색 -> 답변 생성

   1. 문서 관련성 평가 (`grade_documents`):
      - 각 문서의 관련성을 평가
      - 기준을 통과하는 문서만을 유지

   1. 지식 정제 (`refine_knowledge`):
      - 문서를 "지식 조각"으로 분할하고 각각의 관련성을 평가
      - 관련성 높은(0.5 초과) 지식 조각만 유지

   1. 웹 검색 (`web_search`):
      - 문서가 충분한 정보를 담지 못한 경우 외부 지식을 활용
      - 웹 검색 결과를 기존 문서에 추가 

   1. 답변 생성 (`generate_answer`):
      - 정제된 지식 조각을 사용하여 답변을 생성
      - 관련 정보가 없을 경우 적절한 메시지를 반환



###  2-1. Tool 정의

`(1) 벡터저장소 검색기`


```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 인덱스 로드
vector_db = Chroma(
    embedding_function=embeddings_model,   
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)

# 검색도구 생성 
retriever= vector_db.as_retriever(search_kwargs={"k":2})
```

`(2) 웹 검색`


```python
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=2)
```

    /var/folders/vp/t7xb2kg161q5m2ylkq9jn7k00000gn/T/ipykernel_38718/2449524758.py:3: LangChainDeprecationWarning: The class `TavilySearchResults` was deprecated in LangChain 0.3.25 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-tavily package and should be used instead. To use it run `pip install -U :class:`~langchain-tavily` and import as `from :class:`~langchain_tavily import TavilySearch``.
      search_tool = TavilySearchResults(max_results=2)


### 2-2. LLM 모델

`(1) Retrieval Grader`


```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 문서 관련성 평가 결과를 위한 데이터 모델 정의
class GradeDocuments(BaseModel):
    """Three-class score for relevance check on retrieved documents."""

    relevance_score: Literal["correct", "incorrect", "ambiguous"] = Field(
        description="Document relevance to the question: 'correct', 'incorrect', or 'ambiguous'"
    )

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 문서 관련성 평가를 위한 시스템 프롬프트 정의
system_prompt = """
You are an expert evaluator tasked with assessing the relevance of retrieved documents to a user's question. Your role is crucial in enhancing the quality of information retrieval systems.

[평가 기준]
1. 키워드 관련성: 문서가 질문의 주요 단어나 유사어를 포함하는지 확인
2. 의미적 관련성: 문서의 전반적인 주제가 질문의 의도와 일치하는지 평가
3. 부분 관련성: 질문의 일부를 다루거나 맥락 정보를 제공하는 문서도 고려
4. 답변 가능성: 직접적인 답이 아니더라도 답변 형성에 도움될 정보 포함 여부 평가

[점수 체계]
- 'Correct': 문서가 명확히 관련 있고, 질문에 답하는 데 필요한 정보를 포함함.
- 'Incorrect': 문서가 명확히 무관하거나, 질문에 도움이 되지 않는 정보를 포함함.
- 'Ambiguous': 문서의 관련성이 불분명하거나, 일부 관련 정보는 있지만 유용성이 확실하지 않음, 혹은 질문과 약간만 관련 있음.

[주의사항]
- 단순 단어 매칭이 아닌 질문의 전체 맥락을 고려하세요
- 완벽한 답변이 아니어도 유용한 정보가 있다면 관련 있다고 판단하세요

Your evaluation plays a critical role in improving the overall performance of the information retrieval system. Strive for balanced and thoughtful assessments.
"""

# 채점 프롬프트 템플릿 생성
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Document: \n\n {document} \n\n Question: {question}"),
])

# Retrieval Grader 파이프라인 구성
retrieval_grader = grade_prompt | structured_llm_grader
    
# 관련성 평가 실행
question = "비건 메뉴가 있나요?"
retrieved_docs = retriever.invoke(question)
print(f"검색된 문서 수: {len(retrieved_docs)}")

for test_chunk in retrieved_docs:
    print("문서:", test_chunk.page_content)

    relevance = retrieval_grader.invoke({"question": question, "document": test_chunk.page_content})
    print(f"문서 관련성: {relevance.relevance_score}")
    print("=====================================")
```

    검색된 문서 수: 2
    문서: 22. 메인 랍스터 롤
        • 가격: ₩34,000
        • 주요 식재료: 메인 랍스터, 브리오쉬 번, 셀러리, 레몬 마요네즈
        • 설명: 신선한 메인 랍스터를 특제 레몬 마요네즈로 버무려 따뜻한 브리오쉬 번에 듬뿍 채워 넣은 고급 샌드위치입니다. 아삭한 셀러리와 버터 구운 번의 조화가 일품이며, 트러플 감자칩을 곁들입니다.
    문서 관련성: incorrect
    =====================================
    문서: 6. 해산물 파스타
       • 가격: ₩24,000
       • 주요 식재료: 링귀네 파스타, 새우, 홍합, 오징어, 토마토 소스
       • 설명: 알 덴테로 삶은 링귀네 파스타에 신선한 해산물을 듬뿍 올린 메뉴입니다. 토마토 소스의 산미와 해산물의 감칠맛이 조화를 이루며, 마늘과 올리브 오일로 풍미를 더했습니다. 파슬리를 뿌려 향긋한 맛을 더합니다.
    문서 관련성: incorrect
    =====================================



```python
question = "해산물 요리를 추천해주세요."
retrieved_docs = retriever.invoke(question)
print(f"검색된 문서 수: {len(retrieved_docs)}")

for test_chunk in retrieved_docs:
    print("문서:", test_chunk.page_content)

    relevance = retrieval_grader.invoke({"question": question, "document": test_chunk.page_content})
    print(f"문서 관련성: {relevance.relevance_score}")
    print("=====================================")
```

    검색된 문서 수: 2
    문서: 6. 해산물 파스타
       • 가격: ₩24,000
       • 주요 식재료: 링귀네 파스타, 새우, 홍합, 오징어, 토마토 소스
       • 설명: 알 덴테로 삶은 링귀네 파스타에 신선한 해산물을 듬뿍 올린 메뉴입니다. 토마토 소스의 산미와 해산물의 감칠맛이 조화를 이루며, 마늘과 올리브 오일로 풍미를 더했습니다. 파슬리를 뿌려 향긋한 맛을 더합니다.
    문서 관련성: correct
    =====================================
    문서: 30. 씨푸드 빠에야
        • 가격: ₩42,000
        • 주요 식재료: 스페인산 봄바 쌀, 홍합, 새우, 오징어, 사프란
        • 설명: 스페인 전통 방식으로 조리한 해산물 빠에야입니다. 최상급 사프란으로 노란빛을 내고 신선한 해산물을 듬뿍 넣어 지중해의 맛을 그대로 담았습니다. 2인 이상 주문 가능하며 레몬을 곁들여 제공됩니다.
    문서 관련성: correct
    =====================================


`(2) Answer Generator`


```python
# 기본 RAG 체인
from langchain_core.output_parsers import StrOutputParser

def generator_answer(question, docs):

    template = """
    Answer the question based solely on the given context. Do not use any external information or knowledge.

    [Instructions]
        1. Carefully verify information related to the question within the given context.
        2. Use only information directly relevant to the question in your answer.
        3. Do not make assumptions about information not explicitly stated in the context.
        4. Avoid unnecessary information and keep your answer concise and clear.
        5. If an answer cannot be found in the context, respond with "주어진 정보만으로는 답할 수 없습니다."
        6. When appropriate, use direct quotes from the context, using quotation marks.

    [Context]
    {context}

    [Question]
    {question}

    [Answer]
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": format_docs(docs), "question": question})

    return generation


# 검색된 문서를 기반으로 질문에 대한 답변 생성
generation = generator_answer(question, docs=retrieved_docs)
print(generation)
```

    해산물 요리로는 "해산물 파스타"와 "씨푸드 빠에야"가 있습니다.


`(3) Question Re-writer`


```python
def rewrite_question(question: str) -> str:
    """
    주어진 질문을 벡터 저장소 검색에 최적화된 형태로 다시 작성합니다.

    :param question: 원본 질문 문자열
    :return: 다시 작성된 질문 문자열
    """
    # LLM 모델 초기화
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # 시스템 프롬프트 정의
    system_prompt = """
    You are an expert question re-writer. Your task is to convert input questions into optimized versions 
    for vectorstore retrieval. Analyze the input carefully and focus on capturing the underlying semantic 
    intent and meaning. Your goal is to create a question that will lead to more effective and relevant 
    document retrieval.

    [Guidelines]
        1. Identify and emphasize key concepts and main subjects in the question.
        2. Expand abbreviations and clarify ambiguous terms.
        3. Include synonyms or related terms that might appear in relevant documents.
        4. Maintain the original intent and scope of the question.
        5. Break down complex questions into simple, focused sub-questions.

    Remember, the goal is to improve retrieval effectiveness, not to change the fundamental meaning of the question.
    """

    # 질문 다시 쓰기 프롬프트 템플릿 생성
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "[Initial question]\n{question}\n\n[Improved question]\n",
            ),
        ]
    )

    # 질문 다시 쓰기 체인 구성
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # 질문 다시 쓰기 실행
    rewritten_question = question_rewriter.invoke({"question": question})

    return rewritten_question

# 질문 다시 쓰기 테스트
rewritten_question = rewrite_question(question)
print(f"원본 질문: {question}")
print(f"다시 쓴 질문: {rewritten_question}")
```

    원본 질문: 해산물 요리를 추천해주세요.
    다시 쓴 질문: 해산물 요리 추천을 위한 인기 있는 해산물 요리 종류와 조리 방법, 재료별 해산물 요리 레시피를 알려주세요.


`(4) Knowledge Refiner`


```python
# 지식 정제를 위한 클래스
class RefinedKnowledge(BaseModel):
    """
    Represents a refined piece of knowledge extracted from a document.
    """

    knowledge_strip: str = Field(description="A refined piece of knowledge extracted from a document")
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM 모델 초기화 및 구조화된 출력 설정
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
structured_llm_grader = llm.with_structured_output(RefinedKnowledge)

# 지식 정제를 위한 프롬프트
system_prompt = """
    You are an expert in knowledge refinement. Your task is to extract key information from the given document related to the provided question and evaluate its relevance.

    [Instructions]
    1. Carefully read the question and the document.
    2. Identify key pieces of information in the document that are relevant to answering the question.
    3. For each key piece of information:
       a. Extract and summarize it concisely (aim for 1-2 sentences per piece).
       b. Evaluate its relevance to the question as either 'yes' (relevant) or 'no' (not relevant).
    4. Present each piece of information on a new line in the following format:
       [Extracted Information] (yes/no)

    [Example Output]
    AI systems can exhibit biases present in their training data. (yes)
    The use of AI in decision-making raises privacy concerns. (yes)
    Machine learning models require significant computational resources. (no)

    [Note]
    Focus on extracting factual and objective information. Avoid personal opinions or speculations. Aim to provide 3-5 key pieces of information, but you may include more if the document is particularly rich in relevant content.
    """

# 지식정제를 위한 프롬프트 템플릿 생성
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "[Document]\n{document}\n\n[User question]\n{question}"),
])

# Knowledge Refiner 파이프라인 구성
knowledge_refiner = refine_prompt | structured_llm_grader

# 지식 정제 실행
retrieved_docs = retriever.invoke(question)
print(f"검색된 문서 수: {len(retrieved_docs)}")

for test_chunk in retrieved_docs:
    print("문서:", test_chunk.page_content)

    refined_knowledge = knowledge_refiner.invoke({"question": question, "document": test_chunk})
    print(f"정제된 지식: {refined_knowledge.knowledge_strip}")
    print(f"정제된 지식 평가: {refined_knowledge.binary_score}")
    print("=====================================")

```

    검색된 문서 수: 2
    문서: 6. 해산물 파스타
       • 가격: ₩24,000
       • 주요 식재료: 링귀네 파스타, 새우, 홍합, 오징어, 토마토 소스
       • 설명: 알 덴테로 삶은 링귀네 파스타에 신선한 해산물을 듬뿍 올린 메뉴입니다. 토마토 소스의 산미와 해산물의 감칠맛이 조화를 이루며, 마늘과 올리브 오일로 풍미를 더했습니다. 파슬리를 뿌려 향긋한 맛을 더합니다.
    정제된 지식: 해산물 파스타는 링귀네 파스타에 새우, 홍합, 오징어 등 신선한 해산물이 듬뿍 들어간 메뉴입니다.
    정제된 지식 평가: yes
    =====================================
    문서: 30. 씨푸드 빠에야
        • 가격: ₩42,000
        • 주요 식재료: 스페인산 봄바 쌀, 홍합, 새우, 오징어, 사프란
        • 설명: 스페인 전통 방식으로 조리한 해산물 빠에야입니다. 최상급 사프란으로 노란빛을 내고 신선한 해산물을 듬뿍 넣어 지중해의 맛을 그대로 담았습니다. 2인 이상 주문 가능하며 레몬을 곁들여 제공됩니다.
    정제된 지식: 씨푸드 빠에야는 스페인산 봄바 쌀, 홍합, 새우, 오징어, 사프란 등 신선한 해산물을 사용한 스페인 전통 해산물 요리입니다.
    정제된 지식 평가: yes
    =====================================


### 3-3. LangGraph로 그래프 구현

`(1) 그래프 State 생성`


```python
from typing import TypedDict, Union, List, Dict, Tuple, Any
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    그래프의 상태를 나타내는 클래스
    Attributes:
        question: 질문
        generation: LLM 생성 결과
        retrieved_documents: 검색 문서 리스트 (문서, 점수) -> 벡터 저장소 검색 결과 (내부지식)
        knowledge_strips: 지식 보강한 결과 리스트 (문서, 점수) -> 최종 생성에 사용되는 문서 (내부지식 + 외부지식)
        num_generations: 생성 횟수 (무한 루프 방지에 활용)        
    """
    question: str
    generation: str
    retrieved_documents: List[Tuple[Document, str]]
    knowledge_strips: List[Tuple[Document, str]]
    num_generations: int
```

`(2) Node 구성`


```python
def retrieve(state: GraphState) -> GraphState:
    """문서를 검색하는 함수"""
    print("--- 문서 검색 ---")
    question = state["question"]
    
    # 문서 검색 로직
    retrieved_documents = retriever.invoke(question)
    retrieved_documents = [(doc, "ambiguous") for doc in retrieved_documents]

    return {"retrieved_documents": retrieved_documents}


def grade_documents(state: GraphState) -> GraphState:
    """검색된 문서의 관련성을 평가하는 함수"""
    print("--- 문서 관련성 평가 ---")
    question = state["question"]
    retrieved_documents = state.get("retrieved_documents", [])

    # 문서 평가 
    scored_docs = []
    for doc, _ in retrieved_documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = score.relevance_score.lower()
        if grade == "correct":
            print("---문서 관련성: 있음---")
            scored_docs.append((doc, "correct"))

        elif grade == "incorrect":
            print("---문서 관련성: 없음---")
            scored_docs.append((doc, "incorrect"))
        else:
            print("---문서 관련성: 모호함---")
            scored_docs.append((doc, "ambiguous"))

    return {"retrieved_documents": scored_docs}



def refine_knowledge(state: GraphState) -> GraphState:
    """지식을 정제하는 함수"""
    print("--- 지식 정제 ---")
    question = state["question"]
    retrieved_documents = state["retrieved_documents"]
    
    # 지식 정제
    refined_docs = []
    for doc, score in retrieved_documents:

        if score == "incorrect":
            # 관련성이 없는 문서는 제외
            continue

        refined_knowledge = knowledge_refiner.invoke({"question": question, "document": doc.page_content})
        knowledge = refined_knowledge.knowledge_strip
        grade = refined_knowledge.binary_score
        if grade == "yes":
            print("---정제된 지식: 추가---")
            refined_docs.append((Document(page_content=knowledge), "correct"))
        else:
            # 정제된 지식이 없는 경우 제외
            continue
    
    return {"knowledge_strips": refined_docs}



def web_search(state: GraphState) -> GraphState:
    """웹 검색을 수행하는 함수"""
    print("--- 웹 검색 ---")
    question = state["question"]
    
    # 웹 검색 로직
    search_results = search_tool.invoke(question)

    retrieved_documents = [(Document(page_content=str(web_text)), "ambiguous") for web_text in search_results]

    return {"retrieved_documents": retrieved_documents}



def generate(state: GraphState) -> GraphState:
    """답변을 생성하는 함수"""
    print("--- 답변 생성 ---")
    question = state["question"]
    knowledge_strips = state["knowledge_strips"]
    
    # RAG를 이용한 답변 생성
    doc_texts = [doc for doc, _ in knowledge_strips]
    generation = generator_answer(question, docs=doc_texts)

    # 생성 횟수 업데이트
    num_generations = state.get("num_generations", 0)  
    num_generations += 1
    return {"generation": generation, "num_generations": num_generations}


def transform_query(state: GraphState) -> GraphState:
    """질문을 개선하는 함수"""
    print("--- 질문 개선 ---")
    question = state["question"]
    
    # 질문 재작성
    rewritten_question = rewrite_question(question)
    return {"question": rewritten_question}

```

`(3) Edge 구성`


```python
def decide_to_generate(state: GraphState) -> str:
    """답변 생성 여부를 결정하는 함수"""
    print("--- 평가된 문서 분석 ---")
    knowledge_strips = state["knowledge_strips"]
    
    if not knowledge_strips:
        print("--- 결정: 모든 문서가 질문과 관련이 없음, 질문 개선 필요 (-> transform_query)---")
        return "transform_query"
    else:
        print("--- 결정: 답변 생성 (-> generate)---")
        return "generate"
```

`(4) 그래프 연결`


```python
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# 워크플로우 그래프 초기화
builder = StateGraph(GraphState)

# 노드 정의
builder.add_node("retrieve", retrieve)                  # 문서 검색 
builder.add_node("grade_documents", grade_documents)    # 문서 평가
builder.add_node("refine_knowledge", refine_knowledge)  # 지식 정제
builder.add_node("web_search", web_search)              # 웹 검색
builder.add_node("generate", generate)                  # 답변 생성
builder.add_node("transform_query", transform_query)    # 질문 개선


# 경로 정의
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_documents")
builder.add_edge("grade_documents", "refine_knowledge")

# 조건부 엣지 추가: 문서 평가 후 결정
builder.add_conditional_edges(
    "refine_knowledge",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

# 추가 경로 
builder.add_edge("transform_query", "web_search")
builder.add_edge("web_search", "grade_documents")
builder.add_edge("generate", END)


# 그래프 컴파일
graph = builder.compile()

# 그래프 시각화
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_006_LangGraph_CRAG_32_0.png)
    


`(5) 그래프 실행`


```python
inputs = {"question": "스테이크 메뉴의 가격은 얼마인가요?"}

for output in graph.stream(inputs):
    for key, value in output.items():
        # 노드 출력
        pprint(f"Node '{key}':")
        pprint(f"Value: {value}", indent=2, width=80, depth=None)
    print("\n----------------------------------------------------------\n")
```

    --- 문서 검색 ---
    "Node 'retrieve':"
    ("Value: {'retrieved_documents': "
     "[(Document(id='7296141e-87bd-4bce-aab2-6df9a0469569', "
     "metadata={'menu_number': 26, 'source': './data/restaurant_menu.txt', "
     "'menu_name': '샤토브리앙 스테이크'}, page_content='26. 샤토브리앙 스테이크\\n    • 가격: "
     '₩42,000\\n    • 주요 식재료: 프리미엄 안심 스테이크, 푸아그라, 트러플 소스\\n    • 설명: 최상급 안심 스테이크에 '
     '푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리입니다. 부드러운 육질과 깊은 풍미가 특징이며, 그린 아스파라거스와 감자 '
     "그라탕을 함께 제공합니다.'), 'ambiguous'), "
     "(Document(id='4b637c2b-0c05-439d-8ce6-321c6a4fb649', metadata={'source': "
     "'./data/restaurant_menu.txt', 'menu_name': '시그니처 스테이크', 'menu_number': 1}, "
     "page_content='1. 시그니처 스테이크\\n   • 가격: ₩35,000\\n   • 주요 식재료: 최상급 한우 등심, 로즈메리 "
     '감자, 그릴드 아스파라거스\\n   • 설명: 셰프의 특제 시그니처 메뉴로, 21일간 건조 숙성한 최상급 한우 등심을 사용합니다. 미디엄 '
     '레어로 조리하여 육즙을 최대한 보존하며, 로즈메리 향의 감자와 아삭한 그릴드 아스파라거스가 곁들여집니다. 레드와인 소스와 함께 제공되어 '
     "풍부한 맛을 더합니다.'), 'ambiguous')]}")
    
    ----------------------------------------------------------
    
    --- 문서 관련성 평가 ---
    ---문서 관련성: 있음---
    ---문서 관련성: 있음---
    "Node 'grade_documents':"
    ("Value: {'retrieved_documents': "
     "[(Document(id='7296141e-87bd-4bce-aab2-6df9a0469569', "
     "metadata={'menu_number': 26, 'source': './data/restaurant_menu.txt', "
     "'menu_name': '샤토브리앙 스테이크'}, page_content='26. 샤토브리앙 스테이크\\n    • 가격: "
     '₩42,000\\n    • 주요 식재료: 프리미엄 안심 스테이크, 푸아그라, 트러플 소스\\n    • 설명: 최상급 안심 스테이크에 '
     '푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리입니다. 부드러운 육질과 깊은 풍미가 특징이며, 그린 아스파라거스와 감자 '
     "그라탕을 함께 제공합니다.'), 'correct'), "
     "(Document(id='4b637c2b-0c05-439d-8ce6-321c6a4fb649', metadata={'source': "
     "'./data/restaurant_menu.txt', 'menu_name': '시그니처 스테이크', 'menu_number': 1}, "
     "page_content='1. 시그니처 스테이크\\n   • 가격: ₩35,000\\n   • 주요 식재료: 최상급 한우 등심, 로즈메리 "
     '감자, 그릴드 아스파라거스\\n   • 설명: 셰프의 특제 시그니처 메뉴로, 21일간 건조 숙성한 최상급 한우 등심을 사용합니다. 미디엄 '
     '레어로 조리하여 육즙을 최대한 보존하며, 로즈메리 향의 감자와 아삭한 그릴드 아스파라거스가 곁들여집니다. 레드와인 소스와 함께 제공되어 '
     "풍부한 맛을 더합니다.'), 'correct')]}")
    
    ----------------------------------------------------------
    
    --- 지식 정제 ---
    ---정제된 지식: 추가---
    ---정제된 지식: 추가---
    --- 평가된 문서 분석 ---
    --- 결정: 답변 생성 (-> generate)---
    "Node 'refine_knowledge':"
    ("Value: {'knowledge_strips': [(Document(metadata={}, page_content='샤토브리앙 "
     "스테이크의 가격은 ₩42,000입니다.'), 'correct'), (Document(metadata={}, "
     "page_content='시그니처 스테이크의 가격은 ₩35,000입니다.'), 'correct')]}")
    
    ----------------------------------------------------------
    
    --- 답변 생성 ---
    "Node 'generate':"
    ("Value: {'generation': '스테이크 메뉴의 가격은 샤토브리앙 스테이크 ₩42,000, 시그니처 스테이크 "
     "₩35,000입니다.', 'num_generations': 1}")
    
    ----------------------------------------------------------
    



```python
# 최종 답변
print(value["generation"])
```

    스테이크 메뉴의 가격은 샤토브리앙 스테이크 ₩42,000, 시그니처 스테이크 ₩35,000입니다.



```python
inputs = {"question": "스테이크에 어울리는 와인을 추천해주세요."}

for output in graph.stream(inputs):
    for key, value in output.items():
        # 노드 출력
        pprint(f"Node '{key}':")
        pprint(f"Value: {value}", indent=2, width=80, depth=None)
    print("\n----------------------------------------------------------\n")
```

    --- 문서 검색 ---
    "Node 'retrieve':"
    ("Value: {'retrieved_documents': "
     "[(Document(id='a46463e3-23e6-4c69-a9ef-1005b31b401b', metadata={'menu_name': "
     "'안심 스테이크 샐러드', 'menu_number': 8, 'source': './data/restaurant_menu.txt'}, "
     "page_content='8. 안심 스테이크 샐러드\\n   • 가격: ₩26,000\\n   • 주요 식재료: 소고기 안심, 루꼴라, "
     '체리 토마토, 발사믹 글레이즈\\n   • 설명: 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 '
     "샐러드입니다. 체리 토마토와 파마산 치즈 플레이크로 풍미를 더하고, 발사믹 글레이즈로 마무리하여 고기의 풍미를 한층 끌어올렸습니다.'), "
     "'ambiguous'), (Document(id='4b637c2b-0c05-439d-8ce6-321c6a4fb649', "
     "metadata={'source': './data/restaurant_menu.txt', 'menu_name': '시그니처 스테이크', "
     "'menu_number': 1}, page_content='1. 시그니처 스테이크\\n   • 가격: ₩35,000\\n   • 주요 "
     '식재료: 최상급 한우 등심, 로즈메리 감자, 그릴드 아스파라거스\\n   • 설명: 셰프의 특제 시그니처 메뉴로, 21일간 건조 숙성한 '
     '최상급 한우 등심을 사용합니다. 미디엄 레어로 조리하여 육즙을 최대한 보존하며, 로즈메리 향의 감자와 아삭한 그릴드 아스파라거스가 '
     "곁들여집니다. 레드와인 소스와 함께 제공되어 풍부한 맛을 더합니다.'), 'ambiguous')]}")
    
    ----------------------------------------------------------
    
    --- 문서 관련성 평가 ---
    ---문서 관련성: 없음---
    ---문서 관련성: 없음---
    "Node 'grade_documents':"
    ("Value: {'retrieved_documents': "
     "[(Document(id='a46463e3-23e6-4c69-a9ef-1005b31b401b', metadata={'menu_name': "
     "'안심 스테이크 샐러드', 'menu_number': 8, 'source': './data/restaurant_menu.txt'}, "
     "page_content='8. 안심 스테이크 샐러드\\n   • 가격: ₩26,000\\n   • 주요 식재료: 소고기 안심, 루꼴라, "
     '체리 토마토, 발사믹 글레이즈\\n   • 설명: 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 '
     "샐러드입니다. 체리 토마토와 파마산 치즈 플레이크로 풍미를 더하고, 발사믹 글레이즈로 마무리하여 고기의 풍미를 한층 끌어올렸습니다.'), "
     "'incorrect'), (Document(id='4b637c2b-0c05-439d-8ce6-321c6a4fb649', "
     "metadata={'source': './data/restaurant_menu.txt', 'menu_name': '시그니처 스테이크', "
     "'menu_number': 1}, page_content='1. 시그니처 스테이크\\n   • 가격: ₩35,000\\n   • 주요 "
     '식재료: 최상급 한우 등심, 로즈메리 감자, 그릴드 아스파라거스\\n   • 설명: 셰프의 특제 시그니처 메뉴로, 21일간 건조 숙성한 '
     '최상급 한우 등심을 사용합니다. 미디엄 레어로 조리하여 육즙을 최대한 보존하며, 로즈메리 향의 감자와 아삭한 그릴드 아스파라거스가 '
     "곁들여집니다. 레드와인 소스와 함께 제공되어 풍부한 맛을 더합니다.'), 'incorrect')]}")
    
    ----------------------------------------------------------
    
    --- 지식 정제 ---
    --- 평가된 문서 분석 ---
    --- 결정: 모든 문서가 질문과 관련이 없음, 질문 개선 필요 (-> transform_query)---
    "Node 'refine_knowledge':"
    "Value: {'knowledge_strips': []}"
    
    ----------------------------------------------------------
    
    --- 질문 개선 ---
    "Node 'transform_query':"
    ("Value: {'question': '스테이크와 잘 어울리는 와인 종류와 추천 와인 브랜드는 무엇인가요? 스테이크와 와인 페어링에 적합한 "
     "레드 와인 또는 다른 와인 종류의 특징과 맛 프로필도 알려주세요.'}")
    
    ----------------------------------------------------------
    
    --- 웹 검색 ---
    "Node 'web_search':"
    ("Value: {'retrieved_documents': [(Document(metadata={}, "
     'page_content="{\'title\': \'스테이크와 어울리는 최고의 와인: 무엇을 고를 것인가? - 마시자 매거진\', '
     "'url': "
     "'https://mashija.com/%EC%8A%A4%ED%85%8C%EC%9D%B4%ED%81%AC%EC%99%80-%EC%96%B4%EC%9A%B8%EB%A6%AC%EB%8A%94-%EC%B5%9C%EA%B3%A0%EC%9D%98-%EC%99%80%EC%9D%B8-%EB%AC%B4%EC%97%87%EC%9D%84-%EA%B3%A0%EB%A5%BC-%EA%B2%83%EC%9D%B8/', "
     "'content': '<스테이크를 곁들인 레드 와인을 위한 5가지 전형적인 선택\\\\\\\\>\\\\n\\\\n• 카베르네 "
     '소비뇽(Cabernet Sauvignon)  \\\\n• 말벡(Malbec)  \\\\n• 그르나슈/쉬라즈 블렌드(Grenache / '
     'Shiraz blends)  \\\\n• 시라/쉬라즈(Syrah / Shiraz)  \\\\n• '
     '산지오베제(Sangiovese)\\\\n\\\\n육즙이 풍부한 스테이크와 맛있는 와인이 있는 저녁 식사는 적어도 고기 애호가들에게 인생의 '
     '큰 즐거움일 것이다.\\\\n\\\\n와인과 음식 페어링에서 새로운 시도를 하는 것은 항상 재미있지만, 특별한 스테이크 저녁 식사를 '
     '준비할 때 고려해야 할 몇 가지 스타일과 주의사항이 있다.\\\\n\\\\n<스테이크에 곁들이는 레드 '
     '와인\\\\\\\\>\\\\n\\\\n이 포도 품종을 세계 와인 무대에 재등장시키고 고품질 쇠고기에 대한 국가의 명성을 가진 아르헨티나 '
     '덕분에, 말벡 레드 와인은 스테이크와 함께 고전적인 매칭이 되었다. [...] ‘멋지고 생동감 넘치는 카베르네 프랑(Cabernet '
     'Franc)은 어떤가? 아니면 카리냥(Carignan), 쌩소(Cinsault) 또는 서늘한 기후에서 생산한 시라(Syrah)는 어떨까? '
     'DWWA 칠레 지역 의장이자 Decanter Retailer Awards 회장인 리차즈는 “풀바디하지만 우아한 로제(rosé)도 따뜻한 '
     '날에는 잘 어울린다.”라고 말했다.\\\\n\\\\n그는 바디감과 질감이 있지만 스테이크 저녁 식사 중에 미각을 상쾌하게 할 수 있는 '
     '레드 와인을 즐긴다고 말하며, ‘스테이크의 리스크은 ‘무거운 육류 맛 = 무거운 와인’이라고 생각하는 것이다.’라고 '
     '말했다.\\\\n\\\\n– 피노 누아(Pinot Noir)는 스테이크와 어울리는가? –\\\\n\\\\n대부분의 피노 누아 와인은 '
     '스펙트럼의 라이트에서 미디엄 바디에 위치하는 경향이 있으므로, 그 프로필은 종종 더 가벼운 스타일의 육류와 페어링이 주를 이룬다. '
     '[...] ‘와인과 소고기를 페어링하는 가장 쉬운 방법은 소고기와 와인의 풍미 강도를 일치시키는 방법에 대해 생각하는 것이다.’라고 '
     'Hawksmoor 스테이크하우스 레스토랑의 와인 디렉터인 마크 퀵(Mark Quick)이 와인과 쇠고기의 페어링에 관한 심층 기사에서 '
     '언급했다.\\\\n\\\\n예를 들어 고기의 지방 함량을 고려하라. ‘지방이 많을수록 소고기 맛이 더 강해진다.’라고 2020년 '
     '디캔터와의 인터뷰에서 퀵은 언급했다.\\\\n\\\\n다양한 부위의 페어링에 관한 2007년 기사에서 베켓은 상대적으로 지방 함량이 높은 '
     '립아이 스테이크가 넉넉하고 잘 익은 풀바디를 가진 론 북부의 시라(Syrah) 요새의 코뜨 로띠(Côte-Rôtie)와 슈퍼 '
     '투스칸(Super Tuscan)과 잘 어울린다.’라고 언급했다.\\\\n\\\\n베켓은 또한 잘 익은 스테이크에 더 익은 과일 위주의 '
     '스타일의 레드 와인을 추천했다.\\\\n\\\\n– 소스 문제 –\', \'score\': 0.8925721}"), '
     '\'ambiguous\'), (Document(metadata={}, page_content="{\'title\': \'소고기와 잘 '
     "어울리는 와인은 무엇일까요? - 네이버 블로그', 'url': "
     "'https://m.blog.naver.com/brandkim/223544627689', 'content': '와인 페어링에 있어서 레드 "
     '와인은 일반적으로 고기의 단백질 및 지방과 좋은 상호작용을 하는 타닌 프로파일로 인해 소고기와 가장 잘 어울립니다. 대표적인 페어링 '
     '와인으로는 구운 스테이크의 숯불에 잘 어울리는 까베르네 소비뇽과 잘 양념된 소고기와 잘 어울리는 후추 향을 내는 시라가 있습니다. '
     '구이부터 로스팅까지 각 요리 유형에 따라 가장 적합한 와인 스타일이 결정되므로 소고기를 한 입 먹을 때마다 완벽한 와인 한 모금과 함께 '
     '즐길 수 있습니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n와인 페어링의 기초\\\\n\\\\n소고기와 어울리는 와인을 '
     '선택하는 것은 와인과 소고기의 풍미가 어떻게 상호 작용하는지에 대한 깊은 이해에서 시작됩니다. 타닌과 산도 등 와인의 구성과 다양한 '
     '소고기 부위의 맛 프로파일을 어떻게 보완하는지 고려해야 합니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n풍미 프로파일 이해하기 '
     '[...] 레드 와인 쥬: 스테이크에 레드 와인 쥬를 곁들일 때는 쥬와 비슷한 특성을 가진 와인을 선택하는 것이 중요합니다. 카베르네 '
     '소비뇽이나 보르도 와인은 소스의 풍부하고 고소한 향을 잘 살려줄 수 있습니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n치미추리 '
     '소스: 허브와 마늘이 가득한 강렬한 치미추리 소스를 곁들인 스테이크에는 강렬한 풍미를 잘 잡아주고 균형을 잡아줄 수 있는 와인이 '
     '필요합니다. 강렬한 말벡이나 풍미가 강한 템프라니요는 소스의 신선함을 압도하지 않으면서도 조화로운 대조를 '
     '이룹니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n지역별 와인 페어링\\\\n\\\\n소고기와 페어링할 와인을 선택할 때 '
     '지역적 특성은 와인의 구조와 풍미 프로파일에 영향을 미치기 때문에 특정 지역의 와인이 특정 소고기 요리에 더 이상적일 수 '
     "있습니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n소고기와 어울리는 유럽 와인', 'score': "
     '0.8914433}"), \'ambiguous\')]}')
    
    ----------------------------------------------------------
    
    --- 문서 관련성 평가 ---
    ---문서 관련성: 있음---
    ---문서 관련성: 있음---
    "Node 'grade_documents':"
    ("Value: {'retrieved_documents': [(Document(metadata={}, "
     'page_content="{\'title\': \'스테이크와 어울리는 최고의 와인: 무엇을 고를 것인가? - 마시자 매거진\', '
     "'url': "
     "'https://mashija.com/%EC%8A%A4%ED%85%8C%EC%9D%B4%ED%81%AC%EC%99%80-%EC%96%B4%EC%9A%B8%EB%A6%AC%EB%8A%94-%EC%B5%9C%EA%B3%A0%EC%9D%98-%EC%99%80%EC%9D%B8-%EB%AC%B4%EC%97%87%EC%9D%84-%EA%B3%A0%EB%A5%BC-%EA%B2%83%EC%9D%B8/', "
     "'content': '<스테이크를 곁들인 레드 와인을 위한 5가지 전형적인 선택\\\\\\\\>\\\\n\\\\n• 카베르네 "
     '소비뇽(Cabernet Sauvignon)  \\\\n• 말벡(Malbec)  \\\\n• 그르나슈/쉬라즈 블렌드(Grenache / '
     'Shiraz blends)  \\\\n• 시라/쉬라즈(Syrah / Shiraz)  \\\\n• '
     '산지오베제(Sangiovese)\\\\n\\\\n육즙이 풍부한 스테이크와 맛있는 와인이 있는 저녁 식사는 적어도 고기 애호가들에게 인생의 '
     '큰 즐거움일 것이다.\\\\n\\\\n와인과 음식 페어링에서 새로운 시도를 하는 것은 항상 재미있지만, 특별한 스테이크 저녁 식사를 '
     '준비할 때 고려해야 할 몇 가지 스타일과 주의사항이 있다.\\\\n\\\\n<스테이크에 곁들이는 레드 '
     '와인\\\\\\\\>\\\\n\\\\n이 포도 품종을 세계 와인 무대에 재등장시키고 고품질 쇠고기에 대한 국가의 명성을 가진 아르헨티나 '
     '덕분에, 말벡 레드 와인은 스테이크와 함께 고전적인 매칭이 되었다. [...] ‘멋지고 생동감 넘치는 카베르네 프랑(Cabernet '
     'Franc)은 어떤가? 아니면 카리냥(Carignan), 쌩소(Cinsault) 또는 서늘한 기후에서 생산한 시라(Syrah)는 어떨까? '
     'DWWA 칠레 지역 의장이자 Decanter Retailer Awards 회장인 리차즈는 “풀바디하지만 우아한 로제(rosé)도 따뜻한 '
     '날에는 잘 어울린다.”라고 말했다.\\\\n\\\\n그는 바디감과 질감이 있지만 스테이크 저녁 식사 중에 미각을 상쾌하게 할 수 있는 '
     '레드 와인을 즐긴다고 말하며, ‘스테이크의 리스크은 ‘무거운 육류 맛 = 무거운 와인’이라고 생각하는 것이다.’라고 '
     '말했다.\\\\n\\\\n– 피노 누아(Pinot Noir)는 스테이크와 어울리는가? –\\\\n\\\\n대부분의 피노 누아 와인은 '
     '스펙트럼의 라이트에서 미디엄 바디에 위치하는 경향이 있으므로, 그 프로필은 종종 더 가벼운 스타일의 육류와 페어링이 주를 이룬다. '
     '[...] ‘와인과 소고기를 페어링하는 가장 쉬운 방법은 소고기와 와인의 풍미 강도를 일치시키는 방법에 대해 생각하는 것이다.’라고 '
     'Hawksmoor 스테이크하우스 레스토랑의 와인 디렉터인 마크 퀵(Mark Quick)이 와인과 쇠고기의 페어링에 관한 심층 기사에서 '
     '언급했다.\\\\n\\\\n예를 들어 고기의 지방 함량을 고려하라. ‘지방이 많을수록 소고기 맛이 더 강해진다.’라고 2020년 '
     '디캔터와의 인터뷰에서 퀵은 언급했다.\\\\n\\\\n다양한 부위의 페어링에 관한 2007년 기사에서 베켓은 상대적으로 지방 함량이 높은 '
     '립아이 스테이크가 넉넉하고 잘 익은 풀바디를 가진 론 북부의 시라(Syrah) 요새의 코뜨 로띠(Côte-Rôtie)와 슈퍼 '
     '투스칸(Super Tuscan)과 잘 어울린다.’라고 언급했다.\\\\n\\\\n베켓은 또한 잘 익은 스테이크에 더 익은 과일 위주의 '
     '스타일의 레드 와인을 추천했다.\\\\n\\\\n– 소스 문제 –\', \'score\': 0.8925721}"), '
     '\'correct\'), (Document(metadata={}, page_content="{\'title\': \'소고기와 잘 어울리는 '
     "와인은 무엇일까요? - 네이버 블로그', 'url': "
     "'https://m.blog.naver.com/brandkim/223544627689', 'content': '와인 페어링에 있어서 레드 "
     '와인은 일반적으로 고기의 단백질 및 지방과 좋은 상호작용을 하는 타닌 프로파일로 인해 소고기와 가장 잘 어울립니다. 대표적인 페어링 '
     '와인으로는 구운 스테이크의 숯불에 잘 어울리는 까베르네 소비뇽과 잘 양념된 소고기와 잘 어울리는 후추 향을 내는 시라가 있습니다. '
     '구이부터 로스팅까지 각 요리 유형에 따라 가장 적합한 와인 스타일이 결정되므로 소고기를 한 입 먹을 때마다 완벽한 와인 한 모금과 함께 '
     '즐길 수 있습니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n와인 페어링의 기초\\\\n\\\\n소고기와 어울리는 와인을 '
     '선택하는 것은 와인과 소고기의 풍미가 어떻게 상호 작용하는지에 대한 깊은 이해에서 시작됩니다. 타닌과 산도 등 와인의 구성과 다양한 '
     '소고기 부위의 맛 프로파일을 어떻게 보완하는지 고려해야 합니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n풍미 프로파일 이해하기 '
     '[...] 레드 와인 쥬: 스테이크에 레드 와인 쥬를 곁들일 때는 쥬와 비슷한 특성을 가진 와인을 선택하는 것이 중요합니다. 카베르네 '
     '소비뇽이나 보르도 와인은 소스의 풍부하고 고소한 향을 잘 살려줄 수 있습니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n치미추리 '
     '소스: 허브와 마늘이 가득한 강렬한 치미추리 소스를 곁들인 스테이크에는 강렬한 풍미를 잘 잡아주고 균형을 잡아줄 수 있는 와인이 '
     '필요합니다. 강렬한 말벡이나 풍미가 강한 템프라니요는 소스의 신선함을 압도하지 않으면서도 조화로운 대조를 '
     '이룹니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n지역별 와인 페어링\\\\n\\\\n소고기와 페어링할 와인을 선택할 때 '
     '지역적 특성은 와인의 구조와 풍미 프로파일에 영향을 미치기 때문에 특정 지역의 와인이 특정 소고기 요리에 더 이상적일 수 '
     "있습니다.\\\\n\\\\n\\\\u200b\\\\n\\\\n소고기와 어울리는 유럽 와인', 'score': "
     '0.8914433}"), \'correct\')]}')
    
    ----------------------------------------------------------
    
    --- 지식 정제 ---
    ---정제된 지식: 추가---
    ---정제된 지식: 추가---
    --- 평가된 문서 분석 ---
    --- 결정: 답변 생성 (-> generate)---
    "Node 'refine_knowledge':"
    ("Value: {'knowledge_strips': [(Document(metadata={}, page_content='스테이크와 잘 "
     "어울리는 전형적인 레드 와인 종류로는 카베르네 소비뇽, 말벡, 그르나슈/쉬라즈 블렌드, 시라/쉬라즈, 산지오베제가 있다.'), "
     "'correct'), (Document(metadata={}, page_content='레드 와인은 일반적으로 고기의 단백질 및 지방과 "
     "좋은 상호작용을 하는 타닌 프로파일로 인해 소고기, 특히 스테이크와 가장 잘 어울립니다.'), 'correct')]}")
    
    ----------------------------------------------------------
    
    --- 답변 생성 ---
    "Node 'generate':"
    ('Value: {\'generation\': \'스테이크와 잘 어울리는 와인 종류로는 "카베르네 소비뇽, 말벡, 그르나슈/쉬라즈 블렌드, '
     '시라/쉬라즈, 산지오베제"가 있습니다. 레드 와인은 "고기의 단백질 및 지방과 좋은 상호작용을 하는 타닌 프로파일"을 가지고 있어 '
     "소고기, 특히 스테이크와 가장 잘 어울립니다. 추천 와인 브랜드에 대한 정보는 주어지지 않았습니다.', 'num_generations': "
     '1}')
    
    ----------------------------------------------------------
    



```python
# 최종 답변
print(value["generation"])
```

    스테이크와 잘 어울리는 와인 종류로는 "카베르네 소비뇽, 말벡, 그르나슈/쉬라즈 블렌드, 시라/쉬라즈, 산지오베제"가 있습니다. 레드 와인은 "고기의 단백질 및 지방과 좋은 상호작용을 하는 타닌 프로파일"을 가지고 있어 소고기, 특히 스테이크와 가장 잘 어울립니다. 추천 와인 브랜드에 대한 정보는 주어지지 않았습니다.



```python

```
