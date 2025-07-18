#   LangGraph 활용 - 서브 그래프 (Sub-graph)

---

## 환경 설정 및 준비

`(1) Env 환경변수`


```python
from dotenv import load_dotenv
load_dotenv()
```




    True



`(2) 기본 라이브러리`


```python
import os
from glob import glob

from pprint import pprint
import json
```

`(3) langfuase handler 설정`


```python
from langfuse.langchain import CallbackHandler

# 콜백 핸들러 생성
langfuse_handler = CallbackHandler()
```

---

## **Sub-graph (서브그래프)**

- **서브그래프**는 독립적으로 작동하는 하위 그래프 구조를 의미함

- 더 큰 **부모 그래프** 내에서 하나의 노드로 동작하여 모듈화를 실현

- **재사용성**이 높은 컴포넌트를 구축할 수 있어 시스템 확장에 효과적임


- **장점**: 
    - **모듈화**를 통해 코드의 구조가 명확해지고 유지보수가 간편해짐
    - 독립적인 구조로 인해 다양한 시스템에서 **재사용**이 가능함
    - **확장성**이 뛰어나 새로운 기능을 서브그래프 형태로 쉽게 추가 가능
    - 각 서브그래프의 **독립적인 테스트**와 업데이트가 용이함


---

### **[예제 1] 간단한 덧셈, 곱셈 연산을 수행하는 그래프**

1.  **부모 그래프 (Parent Graph):**
    - 사용자로부터 두 개의 숫자를 입력받습니다 (예: `num1`, `num2`).
    - `calculate`라는 서브그래프를 호출합니다.
    *  서브그래프의 결과를 출력합니다

2.  **서브그래프 (Subgraph):**
    - **`add` 노드:** 두 숫자를 더합니다.
    - **`multiply` 노드:** 두 숫자를 곱합니다.
    - 더한 결과와 곱한 결과를 부모 그래프에 반환합니다.

`(1) 서브그래프 정의 (상태 공유)`

- 서브그래프는 부모 그래프와 **상태 공유**를 통해 연결됨
- 서브그래프는 **독립적 실행**이 가능하면서도 부모 상태 접근 가능


```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image, display

# 서브그래프 상태 정의
class SubgraphState(TypedDict):
    num1: int   
    num2: int   
    sum_val: int  # 더하기 결과를 저장할 키 
    product_val: int  # 곱하기 결과를 저장할 키 

# add 노드: 두 숫자를 더하는 함수
def add_node(state: SubgraphState):
    return {"sum_val": state["num1"] + state["num2"]}

# multiply 노드: 두 숫자를 곱하는 함수
def multiply_node(state: SubgraphState):
    return {"product_val": state["num1"] * state["num2"]}

# 서브그래프 빌더 생성
subgraph_builder = StateGraph(SubgraphState)

subgraph_builder.add_node("add", add_node)
subgraph_builder.add_node("multiply", multiply_node)

subgraph_builder.add_edge(START, "add")
subgraph_builder.add_edge("add", "multiply")
subgraph_builder.add_edge("multiply", END) # 서브그래프의 끝을 지정

# 서브그래프 컴파일
subgraph = subgraph_builder.compile()

# 서브그래프 시각화
display(Image(subgraph.get_graph().draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_003_LangGraph_SubGraph_11_0.png)
    


`(2) 부모 그래프 정의 (컴파일된 서브그래프 사용)`

- **ParentState**는 `SubgraphState`와 `num1`, `num2`, `sum_val`, `product_val` **상태 공유**
- `calculate` 노드는 컴파일된 서브그래프를 **포함**


```python
# 부모 그래프 상태 정의
class ParentState(TypedDict):
    num1: int
    num2: int
    sum_val: int
    product_val: int

# 부모 그래프 빌더 생성
parent_builder = StateGraph(ParentState)

# 컴파일된 서브그래프를 'calculate' 노드로 추가
parent_builder.add_node("calculate", subgraph)

parent_builder.add_edge(START, "calculate")
parent_builder.add_edge("calculate", END) # 부모 그래프의 끝

# 부모 그래프 컴파일
parent_graph = parent_builder.compile()

# 부모 그래프 시각화
display(Image(parent_graph.get_graph(xray=True).draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_003_LangGraph_SubGraph_13_0.png)
    


`(3) 그래프 실행`

- 부모그래프를 통해 실행
- 서브그래프는 부모 그래프와 통합되어, 부분 노드로 동작


```python
# 부모 그래프 실행
initial_state = {
    "num1": 5,
    "num2": 3,
}

for event in parent_graph.stream(initial_state, stream_mode=["values", "updates"]):
    pprint(event)
    print("-"*100)
```

    ('values', {'num1': 5, 'num2': 3})
    ----------------------------------------------------------------------------------------------------
    ('updates',
     {'calculate': {'num1': 5, 'num2': 3, 'product_val': 15, 'sum_val': 8}})
    ----------------------------------------------------------------------------------------------------
    ('values', {'num1': 5, 'num2': 3, 'product_val': 15, 'sum_val': 8})
    ----------------------------------------------------------------------------------------------------


`(4) 서브그래프 정의 (상태 변환)`

- 서브그래프와 부모 그래프가 **상태 키 미공유** 시 변환 함수 필요
- 서브그래프 실행 **전후**에 상태 변환 과정 수행


```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 서브그래프 상태 정의 (부모 그래프와 다른 키 사용)
class SubgraphState(TypedDict):
    a: int  # num1 대신 a 사용
    b: int  # num2 대신 b 사용
    sum: int
    product: int

# add, multiply 노드 정의
def add_node(state: SubgraphState):
    return {"sum": state["a"] + state["b"]}

def multiply_node(state: SubgraphState):
    return {"product": state["a"] * state["b"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("add", add_node)
subgraph_builder.add_node("multiply", multiply_node)

subgraph_builder.add_edge(START, "add")
subgraph_builder.add_edge("add", "multiply")
subgraph_builder.add_edge("multiply", END)
subgraph = subgraph_builder.compile()

# 부모 그래프 상태 정의
class ParentState(TypedDict):
    num1: int
    num2: int
    sum_val: int
    product_val: int

# 서브그래프 호출 및 상태 변환 함수
def call_subgraph(state: ParentState):
    # 부모 그래프 상태를 서브그래프 상태로 변환
    subgraph_input = {"a": state["num1"], "b": state["num2"]}
    # 서브그래프 실행
    subgraph_output = subgraph.invoke(subgraph_input)
    # 서브그래프 결과를 부모 그래프 상태로 변환
    return {
        "sum_val": subgraph_output["sum"],
        "product_val": subgraph_output["product"],
    }

# 부모 그래프 빌더 생성
parent_builder = StateGraph(ParentState)
# 서브그래프 호출 함수를 노드로 추가
parent_builder.add_node("calculate", call_subgraph)
parent_builder.add_edge(START, "calculate")
parent_builder.add_edge("calculate", END)
parent_graph = parent_builder.compile()

# 부모 그래프 시각화
display(Image(parent_graph.get_graph(xray=True).draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_003_LangGraph_SubGraph_17_0.png)
    



```python
# 실행 및 결과는 이전과 동일
initial_state = {
    "num1": 5,
    "num2": 3,
}

for event in parent_graph.stream(initial_state, stream_mode=["values", "updates"]):
    pprint(event)
    print("-"*100)
```

    ('values', {'num1': 5, 'num2': 3})
    ----------------------------------------------------------------------------------------------------
    ('updates', {'calculate': {'product_val': 15, 'sum_val': 8}})
    ----------------------------------------------------------------------------------------------------
    ('values', {'num1': 5, 'num2': 3, 'product_val': 15, 'sum_val': 8})
    ----------------------------------------------------------------------------------------------------


---

### **[예제 2] FAQ 챗봇**

`(1) 기본 구조 설정`

- **TypedDict**를 활용하여 메시지와 고객 ID를 포함하는 **공통 상태**를 정의함
- **CustomerServiceState**는 이슈 분류와 만족도 점수를 추가로 관리함
- **FAQState**는 FAQ 매칭 여부와 답변 내용을 확장하여 정의함


```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from IPython.display import Image, display

# 공통 상태 정의 - 상속을 위한 베이스 클래스
class CustomerServiceState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    customer_id: str
    issue_category: str
    satisfaction_score: float    

# FAQ 상태는 CommonState를 상속받아 사용
class FAQState(CustomerServiceState):
    faq_matched: bool
    faq_answer: str
```

`(2) FAQ 처리 서브그래프 구현`

- **check_faq** 함수는 FAQ 매칭 여부를 확인하고 결과를 반환함
- 키워드 기반으로 **FAQ 매칭**을 수행하며 향후 벡터 DB로 확장 가능함
- **respond_faq** 함수는 매칭된 FAQ 답변을 메시지 형태로 변환함


```python
def check_faq(state: FAQState):
    """FAQ 매칭 확인"""
    # customer_id를 사용하여 사용자별 맞춤 FAQ 검색 가능
    last_message = state["messages"][-1]
    customer_id = state["customer_id"]  # CommonState에서 상속받은 속성
    
    print(f"Checking FAQ for customer: {customer_id}")
    
    if "운영시간" in last_message.content:
        return {
            "faq_matched": True,
            "faq_answer": f"안녕하세요 {customer_id}님, 저희 매장은 평일 9시부터 18시까지 운영합니다."
        }
    return {"faq_matched": False}


def respond_faq(state: FAQState):
    """FAQ 응답 생성"""
    return {
        "messages": [AIMessage(content=state["faq_answer"])]
    }

# FAQ 서브그래프 구성
faq_graph = StateGraph(FAQState)
faq_graph.add_node("check_faq", check_faq)
faq_graph.add_node("respond_faq", respond_faq)

# FAQ 그래프 로직 설정
def should_respond_faq(state: FAQState):
    if state["faq_matched"]:
        return "respond_faq"
    return END

faq_graph.add_edge(START, "check_faq")
faq_graph.add_conditional_edges(
    "check_faq", 
    should_respond_faq,
    {
        "respond_faq": "respond_faq",
        END: END
    }
)
faq_graph.add_edge("respond_faq", END)

faq_subgraph = faq_graph.compile()

# FAQ 그래프 시각화
display(Image(faq_subgraph.get_graph().draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_003_LangGraph_SubGraph_23_0.png)
    


`(3) 메인 고객 서비스 그래프 구현`

- **route_inquiry** 함수는 OpenAI를 활용해 고객 문의를 3가지 카테고리로 분류함
- **handle_technical** 함수는 기술 관련 문의를 전문가 연결 방식으로 처리함
- **handle_billing** 함수는 결제 문의에 대해 24시간 이내 응답을 보장함
- 각 처리 함수는 **만족도 점수**를 함께 반환하여 서비스 품질을 측정함


```python
from langchain_openai import ChatOpenAI

# 메인 고객 서비스 로직
def route_inquiry(state: CustomerServiceState):
    """문의 유형 분류"""
    model = ChatOpenAI(model="gpt-4.1-mini")
    last_message = state["messages"][-1]
    response = model.invoke([
        HumanMessage(content=f"""
        다음 고객 문의의 카테고리를 분류해주세요:
        {last_message.content}
        
        [카테고리 선택]:
        - technical: 기술적 문제
        - billing: 결제 문제
        - general: 일반 문의
        """)
    ])
    return {"issue_category": response.content}

def handle_technical(state: CustomerServiceState):
    """기술 문제 처리"""
    print("---Handling technical issue---")
    return {
        "messages": [AIMessage(content="기술 지원팀에 문의가 전달되었습니다. 곧 전문가가 연락드릴 예정입니다.")],
        "satisfaction_score": 0.8   # 실제 구현에서는 고객 피드백을 받아서 설정
    }

def handle_billing(state: CustomerServiceState):
    """결제 문제 처리"""
    print("---Handling billing issue---")
    return {
        "messages": [AIMessage(content="결제 팀에서 확인 후 24시간 이내에 연락드리겠습니다.")],
        "satisfaction_score": 0.7   # 실제 구현에서는 고객 피드백을 받아서 설정
    }

# 메인 그래프 구성
main_graph = StateGraph(CustomerServiceState)

# FAQ 서브그래프를 함수로 래핑
def process_faq(state: CustomerServiceState):
    # 메인 그래프에서 서브그래프 호출 시 CommonState 속성 전달
    faq_state = {
        "messages": state["messages"],
        "customer_id": state["customer_id"],
        "faq_matched": False,
        "faq_answer": ""
    }
    result = faq_subgraph.invoke(faq_state)
    if len(result.get("messages", [])) > 0:
        return {"messages": result["messages"]}
    return state  # FAQ가 없으면 상태 그대로 반환

# 노드 추가
main_graph.add_node("faq", process_faq)     # FAQ 서브그래프 
main_graph.add_node("route", route_inquiry)
main_graph.add_node("technical", handle_technical)
main_graph.add_node("billing", handle_billing)

# 라우팅 로직
def route_to_handler(state: CustomerServiceState):
    if "technical" in state["issue_category"]:
        print("---Routing to technical handler---")
        return "technical"
    elif "billing" in state["issue_category"]:
        print("---Routing to billing handler---")
        return "billing"
    return END

# 엣지 설정
main_graph.add_edge(START, "faq")
main_graph.add_edge("faq", "route")
main_graph.add_conditional_edges(
    "route", 
    route_to_handler,
    {
        "technical": "technical",
        "billing": "billing",
        END: END
    }
    )
main_graph.add_edge("technical", END)
main_graph.add_edge("billing", END)

customer_service = main_graph.compile()

# 메인 그래프 시각화 (xray=True로 하면 하위 그래프도 시각화)
display(Image(customer_service.get_graph(xray=True).draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_003_LangGraph_SubGraph_25_0.png)
    



```python
# FAQ 질문 테스트

# CommonState의 필수 속성들을 포함하여 테스트
faq_input = {
    "messages": [HumanMessage(content="매장 운영시간이 어떻게 되나요?")],
    "customer_id": "user123",  # CommonState 속성
    "issue_category": "",
    "satisfaction_score": 0.0
}
for event in customer_service.stream(faq_input, stream_mode="values"):
    pprint(event)
    print("-" * 200)
```

    {'customer_id': 'user123',
     'issue_category': '',
     'messages': [HumanMessage(content='매장 운영시간이 어떻게 되나요?', additional_kwargs={}, response_metadata={})],
     'satisfaction_score': 0.0}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Checking FAQ for customer: user123
    {'customer_id': 'user123',
     'issue_category': '',
     'messages': [HumanMessage(content='매장 운영시간이 어떻게 되나요?', additional_kwargs={}, response_metadata={}),
                  HumanMessage(content='매장 운영시간이 어떻게 되나요?', additional_kwargs={}, response_metadata={}),
                  AIMessage(content='안녕하세요 user123님, 저희 매장은 평일 9시부터 18시까지 운영합니다.', additional_kwargs={}, response_metadata={})],
     'satisfaction_score': 0.0}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    {'customer_id': 'user123',
     'issue_category': 'general: 일반 문의',
     'messages': [HumanMessage(content='매장 운영시간이 어떻게 되나요?', additional_kwargs={}, response_metadata={}),
                  HumanMessage(content='매장 운영시간이 어떻게 되나요?', additional_kwargs={}, response_metadata={}),
                  AIMessage(content='안녕하세요 user123님, 저희 매장은 평일 9시부터 18시까지 운영합니다.', additional_kwargs={}, response_metadata={})],
     'satisfaction_score': 0.0}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



```python
# 기술 문의 테스트
tech_input = {
    "messages": [HumanMessage(content="로그인이 안 되는데 어떻게 해결하나요?")],
    "customer_id": "user124",  # CommonState 속성
}

for event in customer_service.stream(tech_input, stream_mode="values"):
    pprint(event)
    print("-" * 200)
```

    {'customer_id': 'user124',
     'messages': [HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={})]}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Checking FAQ for customer: user124
    {'customer_id': 'user124',
     'messages': [HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={}),
                  HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={})]}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ---Routing to technical handler---
    {'customer_id': 'user124',
     'issue_category': '[카테고리 선택]:\n- technical: 기술적 문제',
     'messages': [HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={}),
                  HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={})]}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ---Handling technical issue---
    {'customer_id': 'user124',
     'issue_category': '[카테고리 선택]:\n- technical: 기술적 문제',
     'messages': [HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={}),
                  HumanMessage(content='로그인이 안 되는데 어떻게 해결하나요?', additional_kwargs={}, response_metadata={}),
                  AIMessage(content='기술 지원팀에 문의가 전달되었습니다. 곧 전문가가 연락드릴 예정입니다.', additional_kwargs={}, response_metadata={})],
     'satisfaction_score': 0.8}
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



```python

```
