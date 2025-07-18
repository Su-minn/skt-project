# LangGraph 활용 - 상태 그래프 구현

---

## 개념 소개 및 학습 흐름 안내

- **목적:** LangGraph의 상태 기반 그래프 구조와 핵심 개념(StateGraph, Command 등)을 이해하고, 실습을 통해 대화형 워크플로우를 구현하는 방법을 익힙니다.
- **학습 순서:**
  1. StateGraph 기초 개념
  2. 상태/노드/그래프 구조
  3. 실행 방식(invoke, stream)
  4. 조건부 분기 및 Command 활용
  5. 실습
  6. LangGraph Studio 도구 안내

---

## StateGraph

**정의**: 상태 기반의 그래프 구조를 사용하여 대화 흐름을 체계적으로 관리하는 도구입니다.

**특징**:

- 복잡한 대화/업무 흐름을 시각적으로 설계할 수 있습니다.
- 상태(State)를 통해 데이터 흐름을 일관성 있게 관리할 수 있습니다.

**활용**: 복잡한 대화나 업무 흐름을 체계적으로 관리하고 싶을 때 사용합니다.

**관계:**

- **선행 개념**: [Graph (그래프)]
- **연관 개념**: [State (상태), Node (노드), Command]

---

## State (상태)

**정의**: 그래프에서 처리하는 데이터의 기본 구조를 정의하는 요소입니다.

**특징**:

- 각 상태는 다른 상태에 의해 덮어쓰기(override)될 수 있어 데이터를 유연하게 관리할 수 있습니다.
- 상태 관리를 통해 체계적인 데이터 처리와 흐름 제어가 가능합니다.

**활용**: 그래프 내에서 데이터를 저장하고 관리할 때 사용됩니다.

### 코드 예시

```python
from typing import TypedDict

# 상태 정의
class State(TypedDict):
    original_text: str   # 원본 텍스트
    summary: str         # 요약본
    final_summary: str   # 최종 요약본
```

- **설명:** TypedDict를 사용하여 그래프의 상태를 정의합니다. 상태는 그래프 실행 중에 노드 간에 공유되는 데이터입니다.

**관계:**

- **선행 개념**: [TypedDict]
- **연관 개념**: [StateGraph, Node (노드)]

---

## Node (노드)

**정의**: 그래프 구조의 기본 구성 요소로, 독립적인 작업 단위를 나타냅니다.

**특징**:

- 각 노드는 특정 함수를 실행합니다.
- 상태를 입력으로 받아 처리하고 업데이트된 상태를 반환합니다.

**활용**: 그래프 내에서 특정 작업을 수행하는 단위를 정의할 때 사용됩니다.

### 코드 예시

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-mini")
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 요약 생성 노드
def generate_summary(state: State):
    """원본 텍스트를 요약하는 노드"""
    prompt = f"""다음 텍스트를 핵심 내용 중심으로 간단히 요약해주세요:

    [텍스트]
    {state['original_text']}

    [요약]
    """
    response = llm.invoke(prompt)
    return {"summary": response.content}
```

**관계:**

- **선행 개념**: [State (상태)]
- **연관 개념**: [StateGraph, Graph (그래프)]

---

## Graph (그래프) 구성

**정의**: 여러 노드들을 엣지(Edge)로 연결한 집합체입니다.

**특징**:

- 각 노드 간의 연결 관계가 전체 데이터 흐름을 결정합니다.
- 그래프는 빌드가 완료된 후 실행 가능한 상태가 됩니다.

**활용**: 노드들을 연결하여 전체 워크플로우를 구성할 때 사용됩니다.

### 코드 예시

```python
from langgraph.graph import StateGraph, START, END

# StateGraph 객체 생성 (Workflow)
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("summarize", generate_summary)

# 시작(START)과 끝(END) 엣지 추가 : 시작 -> summarize -> 끝
workflow.add_edge(START, "summarize")
workflow.add_edge("summarize", END)

# 그래프 컴파일
graph = workflow.compile()
```

**관계:**

- **선행 개념**: [Node (노드)]
- **연관 개념**: [StateGraph, State (상태)]

---

## 그래프 실행 방식

### invoke 실행

**정의**: 그래프의 가장 기본적인 실행 방법입니다.

**특징**:

- 단순하고 직관적인 처리를 제공합니다.
- 실행 결과는 모든 처리가 완료된 후 최종 결과값만 반환합니다.
- 전체 처리 과정이 완료될 때까지 동기적으로 대기합니다.

**활용**: 간단한 워크플로우를 실행하고 최종 결과를 얻고자 할 때 사용됩니다.

#### 코드 예시

```python
from pprint import pprint

# 사용 예시
text = """
인공지능(AI)은 컴퓨터 과학의 한 분야로, 인간의 학습능력과 추론능력, 지각능력,
자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다.
최근에는 기계학습과 딥러닝의 발전으로 다양한 분야에서 활용되고 있다.
"""

initial_state = {
    "original_text": text,
}

final_state = graph.invoke(initial_state)

for key, value in final_state.items():
    print(f"{key}")
    print("-" * 50)
    pprint(f"{value}")
    print("=" * 100)
```

**관계:**

- **선행 개념**: [Graph (그래프), State (상태)]
- **연관 개념**: [stream 실행]

---

## 조건부 Edge (엣지)

**정의**: 노드 간의 연결 경로를 정의하며, 조건에 따라 다른 경로로 분기할 수 있는 엣지입니다.

**특징**:

- 상황에 따라 다른 경로로 분기할 수 있어 유연한 대화 구조가 가능합니다.
- 사용자의 입력이나 상태에 따라 동적으로 경로가 결정되어 맥락에 맞는 응답을 제공합니다.

**활용**: 조건에 따라 워크플로우의 흐름을 변경해야 할 때 사용됩니다.

### 코드 예시

```python
from typing import Literal

# 요약 품질 체크 노드 (조건부 엣지와 함께 사용)
def check_summary_quality(state: State) -> Literal["needs_improvement", "good"]:
    """요약의 품질을 체크하고 개선이 필요한지 판단하는 노드"""
    prompt = f"""다음 요약의 품질을 평가해주세요.
    원문을 잘 요약했는지 평가해주세요. 요약이 명확하고 핵심을 잘 전달하면 'good'을,
    개선이 필요하면 'needs_improvement'를 응답해주세요.

    원문: {state['original_text']}

    요약본: {state['summary']}
    """
    response = llm.invoke(prompt).content.lower().strip()

    if "good" in response:
        print("---- Good Summary ----")
        return "good"
    else:
        print("---- Needs Improvement ----")
        return "needs_improvement"

# 요약 개선 노드
def improve_summary(state: State):
    """요약을 개선하고 다듬는 노드"""
    prompt = f"""다음 요약을 더 명확하고 간결하게 개선해주세요:

    요약본: {state['summary']}
    """
    response = llm.invoke(prompt)
    return {"final_summary": response.content}

# 요약 완료 노드
def finalize_summary(state: State):
    """현재 요약을 최종 요약으로 설정하는 노드"""
    return {"final_summary": state["summary"]}
```

#### 워크플로우 구성 및 실행

- **목적:** 조건부 분기를 활용한 워크플로우 설계 및 실행 방법을 익힙니다.
- **구성:** START → generate_summary → (조건부 분기) → improve_summary/finalize_summary → END

```python
# 워크플로우 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("improve_summary", improve_summary)
workflow.add_node("finalize_summary", finalize_summary)

# 조건부 엣지 추가를 위한 라우팅 설정
workflow.add_conditional_edges(
    "generate_summary",
    check_summary_quality,
    {
        "needs_improvement": "improve_summary",
        "good": "finalize_summary"
    }
)

# 기본 엣지 추가
workflow.add_edge(START, "generate_summary")
workflow.add_edge("improve_summary", END)
workflow.add_edge("finalize_summary", END)

# 그래프 컴파일
graph = workflow.compile()
```

**관계:**

- **선행 개념**: [Node (노드), Graph (그래프), State (상태)]
- **연관 개념**: [Command]
- **반대 개념**: [Command]

---

## stream 실행

**정의**: 그래프 실행의 중간 과정을 실시간으로 확인할 수 있는 방식입니다.

**특징**:

- 각 노드의 실행 결과가 순차적으로 스트리밍되어 처리 흐름을 자세히 모니터링할 수 있습니다.
- 실시간 피드백이 필요한 복잡한 처리나 사용자 상호작용이 중요한 경우에 적합합니다.

**활용**: 복잡한 워크플로우의 진행 상황을 실시간으로 모니터링하거나 사용자에게 중간 결과를 보여주고 싶을 때 사용됩니다.

### stream_mode 옵션

- **"values"**: 상태 값의 변경사항만 스트리밍
- **"updates"**: 어떤 노드가 업데이트를 생성했는지 포함 (디버깅용)
- **"all"**: 각 업데이트의 타입과 내용을 튜플로 반환 (가장 상세)

#### 코드 예시

```python
# 1. "values" 모드 : 상태 값의 변경사항만 스트리밍
for chunk in graph.stream(initial_state, stream_mode="values"):
    print(chunk)
    print("=" * 100)
```

```python
# 2. "updates" 모드 : 어떤 노드가 업데이트를 생성했는지 포함 (디버깅용)
for chunk in graph.stream(initial_state, stream_mode="updates"):
    print(chunk)
    print("=" * 100)
```

```python
# 3. "all" 모드 : 각 업데이트의 타입과 내용을 튜플로 반환 (가장 상세)
for chunk_type, chunk_data in graph.stream(initial_state, stream_mode=["values", "updates"]):
    print(f"업데이트 타입: {chunk_type}")
    print(f"데이터: {chunk_data}")
    print("=" * 100)
```

**관계:**

- **선행 개념**: [Graph (그래프), State (상태)]
- **연관 개념**: [invoke 실행]

---

## [실습] 조건부 라우팅 StateGraph 구현

### 목적

- 사용자의 언어(한국어/영어)를 감지하여, 한국어 질문이면 한국어 DB를, 영어 질문이면 영어 DB를 검색하는 라우팅 기능을 StateGraph로 구현합니다.
- 리비안, 테슬라 데이터베이스를 사용합니다.

### 벡터 DB 준비

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# OpenAI 임베딩 모델 생성
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")

# 한국어 문서로 저장되어 있는 벡터 저장소 로드
db_korean = Chroma(
    embedding_function=embeddings_openai,
    collection_name="db_korean_cosine_metadata",
    persist_directory="./chroma_db",
    )

print(f"한국어 문서 수: {db_korean._collection.count()}")

# 영어 문서를 저장하는 벡터 저장소 로드
db_english = Chroma(
    embedding_function=embeddings_openai,
    collection_name="eng_db_openai",
    persist_directory="./chroma_db",
    )

print(f"영어 문서 수: {db_english._collection.count()}")
```

### 그래프 및 노드 구현

```python
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-mini")

# state 스키마
class EelectricCarState(TypedDict):
    user_query: str
    is_korean: bool
    search_results: List[str]
    final_answer: str

def analyze_input(state: EelectricCarState) -> EelectricCarState:
    """ 사용자 입력이 한국어인지 판단하는 함수 """

    # 사용자 의도를 분석하기 위한 템플릿
    analyze_template = """
    사용자의 입력을 분석하여 한국어인지 판단하세요.

    사용자 입력: {user_query}

    한국어인 경우 "True", 아니면 "False"로 답변하세요.

    답변:
    """

    # 사용자 입력을 분석하여 한국어인지 판단
    analyze_prompt = ChatPromptTemplate.from_template(analyze_template)
    analyze_chain = analyze_prompt | llm | StrOutputParser()
    result = analyze_chain.invoke({"user_query": state['user_query']})
    is_korean = result.strip().lower() == "true"

    # 결과를 상태에 업데이트
    return {"is_korean": is_korean}


def korean_rag_search(state: EelectricCarState) -> EelectricCarState:
    """ 한국어 문서 검색 함수 """

    # 3개의 검색 결과를 가져옴
    results = db_korean.similarity_search(state['user_query'], k=3)

    # 페이지 내용만 추출
    search_results = [doc.page_content for doc in results]

    # 검색 결과를 상태에 저장
    return {"search_results": search_results}


def english_rag_search(state: EelectricCarState) -> EelectricCarState:
    """ 영어 문서 검색 함수 """

    # 3개의 검색 결과를 가져옴
    results = db_english.similarity_search(state['user_query'], k=3)

    # 페이지 내용만 추출
    search_results = [doc.page_content for doc in results]

    # 검색 결과를 상태에 저장
    return {"search_results": search_results}


def generate_response(state: EelectricCarState) -> EelectricCarState:
    """ 답변 생성 함수 """

    # 답변 템플릿
    response_template = """
    사용자 입력: {user_query}
    검색 결과: {search_results}

    위 정보를 바탕으로 사용자의 질문에 대한 상세한 답변을 생성하세요.
    검색 결과의 정보를 활용하여 정확하고 유용한 정보를 제공하세요.

    답변 (Answer in {language}):
    """

    # 답변 생성
    response_prompt = ChatPromptTemplate.from_template(response_template)
    response_chain = response_prompt | llm | StrOutputParser()

    final_answer = response_chain.invoke(
        {
            "user_query": state['user_query'],   # 사용자 입력 (상태에서 가져옴)
            "search_results": state['search_results'],  # 검색 결과 (상태에서 가져옴)
            "language": "한국어" if state['is_korean'] else "영어" # 언어 정보
        }
    )

    # 결과를 상태에 저장
    return {"final_answer": final_answer}

def decide_next_step(
        state: EelectricCarState
    ) -> Literal["korean_rag_search", "english_rag_search"]:
    """ 다음 실행 단계를 결정하는 함수 """

    # 한국어인 경우 한국어 문서 검색 함수 실행
    if state['is_korean']:
        return "korean"

    # 영어인 경우 영어 문서 검색 함수 실행
    else:
        return "english"

# 그래프 구성
builder = StateGraph(EelectricCarState)

# 노드 추가: 입력 분석, 한국어 문서 검색, 영어 문서 검색, 답변 생성
builder.add_node("analyze_input", analyze_input)
builder.add_node("korean_rag_search", korean_rag_search)
builder.add_node("english_rag_search", english_rag_search)
builder.add_node("generate_response", generate_response)

# 엣지 추가: 시작 -> 사용자 입력 분석 -> 조건부 엣지 -> 한국어 문서 검색 또는 영어 문서 검색 -> 답변 생성 -> 끝
builder.add_edge(START, "analyze_input")

# 조건부 엣지 추가
builder.add_conditional_edges(
    "analyze_input",
    decide_next_step,
    {
        "korean": "korean_rag_search",
        "english": "english_rag_search"
    }
)

builder.add_edge("korean_rag_search", "generate_response")
builder.add_edge("english_rag_search", "generate_response")
builder.add_edge("generate_response", END)

# 그래프 컴파일
graph = builder.compile()
```

#### 실행 예시

```python
# 그래프 실행을 위한 상태 초기화 (한국어 질문)
initial_state = {'user_query':'테슬라의 창업자는 누구인가요?'}

# 그래프 실행
result = graph.invoke(initial_state)

# 결과 출력
print("\n=== 결과 ===")
print("사용자 입력:", initial_state['user_query'])
print("답변:", result['final_answer'])
```

```python
# 그래프 실행을 위한 상태 초기화 (영어 질문)
initial_state = {'user_query':'Who is the founder of Tesla?'}
result = graph.invoke(initial_state)

# 결과 출력
print("\n=== 결과 ===")
print("사용자 입력:", initial_state['user_query'])
print("답변:", result['final_answer'])
```

**관계:**

- **선행 개념**: [StateGraph, Node (노드), 조건부 Edge (엣지)]
- **연관 개념**: [Command]

---

## Command

**정의**: LangGraph의 핵심 제어 도구로, 노드 함수의 반환값으로 사용되어 상태 관리와 흐름 제어를 동시에 수행합니다.

**특징**:

- 상태 업데이트: 그래프의 상태(State)를 변경할 수 있습니다. (예: 새로운 정보를 추가하거나 기존 정보를 수정)
- 제어 흐름: 다음에 실행할 노드를 지정할 수 있습니다.

**활용**: 그래프의 상태를 업데이트하고 다음 실행할 노드를 결정해야 할 때 사용됩니다.

**관계:**

- **선행 개념**: [Node (노드), State (상태)]
- **연관 개념**: [StateGraph, 조건부 Edge (엣지)]

---

### StateGraph에서 Command 활용

#### 구조 및 구현 세부사항

1. `check_summary_quality` 함수를 제거하고 해당 로직을 `generate_summary` 노드에 통합
2. 각 노드가 `Command` 객체를 반환하도록 수정
   - `goto`: 다음에 실행할 노드 지정
   - `update`: 상태 업데이트 내용 지정
3. 조건부 엣지 대신 `Command`를 통한 라우팅을 사용
   - 품질 평가에 따라 `generate_summary`에서 `improve_summary` 또는 `finalize_summary`로 라우팅
   - `improve_summary`와 `finalize_summary`는 모두 END로 라우팅
4. 엣지 설정을 단순화
   - START에서 `generate_summary`로의 엣지만 명시적으로 정의
   - 나머지 라우팅은 `Command`를 통해 처리

#### 코드 예시

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI

# 상태 정의
class State(TypedDict):
    original_text: str   # 원본 텍스트
    summary: str         # 요약본
    final_summary: str   # 최종 요약본

# LLM 인스턴스 생성
summary_llm = ChatOpenAI(model="gpt-4.1-mini")
eval_llm = ChatOpenAI(model="gpt-4.1")

# 요약 생성 노드
def generate_summary(state: State) -> Command[Literal["improve_summary", "finalize_summary"]]:
    """원본 텍스트를 요약하고 품질을 평가하는 노드"""
    # 요약 생성
    summary_prompt = f"""다음 텍스트를 핵심 내용 중심으로 간단히 요약해주세요:

    [텍스트]
    {state['original_text']}

    [요약]
    """
    summary = summary_llm.invoke(summary_prompt).content

    # 품질 평가
    eval_prompt = f"""다음 요약의 품질을 평가해주세요.
    요약이 명확하고 핵심을 잘 전달하면 'good'을,
    개선이 필요하면 'needs_improvement'를 응답해주세요.

    요약본: {summary}
    """
    quality = eval_llm.invoke(eval_prompt).content.lower().strip()

    # 상태 업데이트와 함께 다음 노드로 라우팅
    return Command(
        goto="finalize_summary" if "good" in quality else "improve_summary",
        update={"summary": summary}
    )

# 요약 개선 노드
def improve_summary(state: State) -> Command[Literal[END]]:
    """요약을 개선하고 다듬는 노드"""
    prompt = f"""다음 요약을 더 명확하고 간결하게 개선해주세요:

    [기존 요약]
    {state['summary']}

    [개선 요약]
    """
    improved_summary = summary_llm.invoke(prompt).content

    # 상태 업데이트와 함께 다음 노드로 라우팅
    return Command(
        goto=END,
        update={"final_summary": improved_summary}
    )

# 최종 요약 설정 노드
def finalize_summary(state: State) -> Command[Literal[END]]:
    """현재 요약을 최종 요약으로 설정하는 노드"""

    # 상태 업데이트와 함께 다음 노드로 라우팅
    return Command(
        goto=END,
        update={"final_summary": state["summary"]}
    )
```

**관계:**

- **선행 개념**: [Command, StateGraph, Node (노드), State (상태)]
- **연관 개념**: [조건부 Edge (엣지)]

---

### Command vs. 조건부 엣지(Conditional Edges)

- **Command**: 상태 업데이트와 노드 이동을 동시에 처리할 때 사용. 정보 전달이 필요한 복잡한 전환에 적합.
- **조건부 엣지**: 단순한 분기 처리에 사용. 상태 변경 없이 조건에 따른 이동만 수행.
- **선택 기준:** 상태 업데이트 필요 여부에 따라 결정.

**관계:**

- **연관 개념**: [Command, 조건부 Edge (엣지)]
- **반대 개념**: [조건부 Edge (엣지)]

---

### Command 기반 워크플로우 구성 및 실행

#### 그래프 구성

```python
# 워크플로우 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("improve_summary", improve_summary)
workflow.add_node("finalize_summary", finalize_summary)

# 기본 엣지 추가
workflow.add_edge(START, "generate_summary")

# 그래프 컴파일
graph = workflow.compile()
```

#### 실행 및 결과 확인

```python
from pprint import pprint

# 그래프 실행 및 결과 확인
text = """
인공지능(AI)은 컴퓨터 과학의 한 분야로, 인간의 학습능력과 추론능력, 지각능력,
자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술이다.
최근에는 기계학습과 딥러닝의 발전으로 다양한 분야에서 활용되고 있다.
"""

initial_state = {
    "original_text": text,
}

for chunk in graph.stream(initial_state, stream_mode="values"):
    pprint(chunk)
    print("=" * 100)
```

**관계:**

- **선행 개념**: [Command, StateGraph, Node (노드), State (상태)]

---

## [실습] Command 구문 기반 라우팅 StateGraph 구현

### 목적

- 사용자의 언어를 감지하여, 한국어 질문이면 한국어 DB를, 영어 질문이면 영어 DB를 검색하는 라우팅 기능을 StateGraph로 구현합니다.
- 리비안, 테슬라 데이터베이스를 사용합니다.
- **Command 구문을 사용합니다.**

### 벡터 DB 준비

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# OpenAI 임베딩 모델 생성
embeddings_openai = OpenAIEmbeddings(model="text-embedding-3-small")

# 한국어 문서로 저장되어 있는 벡터 저장소 로드
db_korean = Chroma(
    embedding_function=embeddings_openai,
    collection_name="db_korean_cosine_metadata",
    persist_directory="./chroma_db",
    )

print(f"한국어 문서 수: {db_korean._collection.count()}")

# 영어 문서를 저장하는 벡터 저장소 로드
db_english = Chroma(
    embedding_function=embeddings_openai,
    collection_name="eng_db_openai",
    persist_directory="./chroma_db",
    )

print(f"영어 문서 수: {db_english._collection.count()}")
```

### Command 기반 그래프 및 노드 구현

```python
from typing import List, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import Command

# LLM 인스턴스 생성
llm = ChatOpenAI(model="gpt-4.1-mini")

# state 스키마
class ElectricCarState(TypedDict):
    user_query: str
    is_korean: bool
    search_results: List[str]
    final_answer: str

def analyze_input(state: ElectricCarState) -> Command[Literal["korean_rag_search", "english_rag_search"]]:
    """ 사용자 입력이 한국어인지 판단하고 다음 단계를 Command로 지정하는 함수 """

    # 사용자 의도를 분석하기 위한 템플릿
    analyze_template = """
    사용자의 입력을 분석하여 한국어인지 판단하세요.

    사용자 입력: {user_query}

    한국어인 경우 "True", 아니면 "False"로 답변하세요.

    답변:
    """

    # 사용자 입력을 분석하여 한국어인지 판단
    analyze_prompt = ChatPromptTemplate.from_template(analyze_template)
    analyze_chain = analyze_prompt | llm | StrOutputParser()
    result = analyze_chain.invoke({"user_query": state['user_query']})
    is_korean = result.strip().lower() == "true"

    # Command를 사용하여 다음 노드와 상태 업데이트를 지정
    next_node = "korean_rag_search" if is_korean else "english_rag_search"

    return Command(
        goto=next_node,  # 다음에 실행할 노드 지정
        update={"is_korean": is_korean}  # 상태 업데이트
    )


def korean_rag_search(state: ElectricCarState) -> Command[Literal["generate_response"]]:
    """ 한국어 문서 검색 함수 """

    # 3개의 검색 결과를 가져옴
    results = db_korean.similarity_search(state['user_query'], k=3)

    # 페이지 내용만 추출
    search_results = [doc.page_content for doc in results]

    # Command를 사용하여 다음 노드와 상태 업데이트를 지정
    return Command(
        goto="generate_response",  # 항상 generate_response로 이동
        update={"search_results": search_results}  # 검색 결과 업데이트
    )


def english_rag_search(state: ElectricCarState) -> Command[Literal["generate_response"]]:
    """ 영어 문서 검색 함수 """

    # 3개의 검색 결과를 가져옴
    results = db_english.similarity_search(state['user_query'], k=3)

    # 페이지 내용만 추출
    search_results = [doc.page_content for doc in results]

    # Command를 사용하여 다음 노드와 상태 업데이트를 지정
    return Command(
        goto="generate_response",  # 항상 generate_response로 이동
        update={"search_results": search_results}  # 검색 결과 업데이트
    )

def generate_response(state: ElectricCarState) -> Command[Literal[END]]:
    """ 답변 생성 함수 """

    # 답변 템플릿
    response_template = """
    사용자 입력: {user_query}
    검색 결과: {search_results}

    위 정보를 바탕으로 사용자의 질문에 대한 상세한 답변을 생성하세요.
    검색 결과의 정보를 활용하여 정확하고 유용한 정보를 제공하세요.

    답변 (Answer in {language}):
    """

    # 답변 생성
    response_prompt = ChatPromptTemplate.from_template(response_template)
    response_chain = response_prompt | llm | StrOutputParser()

    final_answer = response_chain.invoke(
        {
            "user_query": state['user_query'],
            "search_results": state['search_results'],
            "language": "한국어" if state['is_korean'] else "영어"
        }
    )

    # Command를 사용하여 END로 이동하고 최종 답변 업데이트
    return Command(
        goto=END,  # 그래프 실행 종료
        update={"final_answer": final_answer}  # 최종 답변 저장
    )

# 그래프 구성
builder = StateGraph(ElectricCarState)

# 노드 추가: 입력 분석, 한국어 문서 검색, 영어 문서 검색, 답변 생성
builder.add_node("analyze_input", analyze_input)
builder.add_node("korean_rag_search", korean_rag_search)
builder.add_node("english_rag_search", english_rag_search)
builder.add_node("generate_response", generate_response)

# 엣지 추가: 시작 -> 사용자 입력 분석
builder.add_edge(START, "analyze_input")

# 그래프 컴파일
graph = builder.compile()
```

#### 실행 예시

```python
# 그래프 실행을 위한 상태 초기화 (한국어 질문)
initial_state = {'user_query':'테슬라의 창업자는 누구인가요?'}

# 그래프 실행
result = graph.invoke(initial_state)

# 결과 출력
print("\n=== 결과 ===")
print("사용자 입력:", initial_state['user_query'])
print("답변:", result['final_answer'])
```

```python
# 그래프 실행을 위한 상태 초기화 (영어 질문)
initial_state = {'user_query':'Who is the founder of Tesla?'}
result = graph.invoke(initial_state)

# 결과 출력
print("\n=== 결과 ===")
print("사용자 입력:", initial_state['user_query'])
print("답변:", result['final_answer'])
```

**관계:**

- **선행 개념**: [Command, StateGraph, Node (노드), State (상태)]

---

## LangGraph Studio

**정의**: 다중 에이전트 워크플로우를 구축하고 테스트할 수 있는 플랫폼입니다.

**특징**:

- 로컬 개발 환경에서 구축한 워크플로우를 시각적으로 테스트하고 디버깅할 수 있습니다.

**활용**: LangGraph 워크플로우를 개발하고 테스트할 때 사용됩니다.

**관계:**

- **선행 개념**: [StateGraph, Node (노드), State (상태), Command]
- **연관 개념**: [LangSmith]

---

### 설치 및 설정

#### 필수 요구사항

- Python 3.11 이상
- LangSmith API 키 (무료 가입 가능)

#### LangGraph CLI 설치

```bash
# Python 서버용 CLI 설치
pip install --upgrade "langgraph-cli[inmem]"

# uv 환경에서 설치
uv add "langgraph-cli[inmem]"
```

#### langgraph.json 파일(프로젝트 폴더)

```json
{
  "dependencies": ["."],
  "graphs": {
    "main_agent": "./src/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.12"
}
```

---

### 개발 서버 실행

- 프로젝트 루트 디렉토리에서 실행

```bash
# 기본 실행
langgraph dev

# 디버깅 모드 (선택적)
langgraph dev --debug-port 5678

# Safari 사용자의 경우
langgraph dev --tunnel
```

---

### LangGraph Studio 접속

- **방법 1: 직접 URL 접속**

  ```
  https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
  ```

- **방법 2: LangSmith UI를 통한 접속**
  1. https://smith.langchain.com 로그인
  2. "LangGraph Platform Deployments" 탭 이동
  3. "LangGraph Studio" 버튼 클릭
  4. `http://127.0.0.1:2024` 입력 후 "Connect"

---
