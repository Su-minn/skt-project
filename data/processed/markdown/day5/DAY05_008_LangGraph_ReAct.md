#   LangGraph 활용 - ReAct 에이전트 활용

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

`(3) Langsmith tracing 설정`


```python
# Langsmith tracing 여부를 확인 (true: langsmith 추적 활성화, false: langsmith 추적 비활성화)
import os
print(os.getenv('LANGSMITH_TRACING'))
```

    true


---

## **레스토랑 메뉴 DB**


`(1) 문서 로드`


```python
from langchain.document_loaders import TextLoader
import re

# 메뉴판 텍스트 데이터를 로드
loader = TextLoader("./data/restaurant_menu.txt", encoding="utf-8")
documents = loader.load()

print(len(documents))
from langchain_core.documents import Document

# 문서 분할 (Chunking)
def split_menu_items(document):
    """
    메뉴 항목을 분리하는 함수 
    """
    # 정규표현식 정의 
    pattern = r'(\d+\.\s.*?)(?=\n\n\d+\.|$)'
    menu_items = re.findall(pattern, document.page_content, re.DOTALL)
    
    # 각 메뉴 항목을 Document 객체로 변환
    menu_documents = []
    for i, item in enumerate(menu_items, 1):
        # 메뉴 이름 추출
        menu_name = item.split('\n')[0].split('.', 1)[1].strip()
        
        # 새로운 Document 객체 생성
        menu_doc = Document(
            page_content=item.strip(),
            metadata={
                "source": document.metadata['source'],
                "menu_number": i,
                "menu_name": menu_name
            }
        )
        menu_documents.append(menu_doc)
    
    return menu_documents


# 메뉴 항목 분리 실행
menu_documents = []
for doc in documents:
    menu_documents += split_menu_items(doc)

# 결과 출력
print(f"총 {len(menu_documents)}개의 메뉴 항목이 처리되었습니다.")
for doc in menu_documents[:2]:
    print(f"\n메뉴 번호: {doc.metadata['menu_number']}")
    print(f"메뉴 이름: {doc.metadata['menu_name']}")
    print(f"내용:\n{doc.page_content[:100]}...")
```

    1
    총 30개의 메뉴 항목이 처리되었습니다.
    
    메뉴 번호: 1
    메뉴 이름: 시그니처 스테이크
    내용:
    1. 시그니처 스테이크
       • 가격: ₩35,000
       • 주요 식재료: 최상급 한우 등심, 로즈메리 감자, 그릴드 아스파라거스
       • 설명: 셰프의 특제 시그니처 메뉴로, ...
    
    메뉴 번호: 2
    메뉴 이름: 트러플 리조또
    내용:
    2. 트러플 리조또
       • 가격: ₩22,000
       • 주요 식재료: 이탈리아산 아르보리오 쌀, 블랙 트러플, 파르미지아노 레지아노 치즈
       • 설명: 크리미한 텍스처의 리조...



```python
# 와인 메뉴 텍스트를 로드
wine_loader = TextLoader("./data/restaurant_wine.txt", encoding="utf-8")

# 와인 메뉴 문서 생성
wine_docs = wine_loader.load()

# 와인 메뉴 문서 분할
wine_documents = []
for doc in wine_docs:
    wine_documents += split_menu_items(doc)

# 결과 출력
print(f"총 {len(wine_documents)}개의 와인 메뉴 항목이 처리되었습니다.")
for doc in wine_documents[:2]:
    print(f"\n메뉴 번호: {doc.metadata['menu_number']}")
    print(f"메뉴 이름: {doc.metadata['menu_name']}")
    print(f"내용:\n{doc.page_content[:100]}...")
```

    총 20개의 와인 메뉴 항목이 처리되었습니다.
    
    메뉴 번호: 1
    메뉴 이름: 샤토 마고 2015
    내용:
    1. 샤토 마고 2015
       • 가격: ₩450,000
       • 주요 품종: 카베르네 소비뇽, 메를로, 카베르네 프랑, 쁘띠 베르도
       • 설명: 보르도 메독 지역의 프리미엄 ...
    
    메뉴 번호: 2
    메뉴 이름: 돔 페리뇽 2012
    내용:
    2. 돔 페리뇽 2012
       • 가격: ₩380,000
       • 주요 품종: 샤르도네, 피노 누아
       • 설명: 프랑스 샴페인의 대명사로 알려진 프레스티지 큐베입니다. 시트러스...


`(2) 벡터스토어 저장`


```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩 모델 생성
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 메뉴판 Chroma 인덱스 생성
menu_db = Chroma.from_documents(
    documents=menu_documents, 
    embedding=embeddings_model,   
    collection_name="restaurant_menu",
    persist_directory="./chroma_db",
)

# 와인 메뉴 Chroma 인덱스 생성
wine_db = Chroma.from_documents(
    documents=wine_documents, 
    embedding=embeddings_model,   
    collection_name="restaurant_wine",
    persist_directory="./chroma_db",
)
```

`(3) 벡터 검색기 테스트`


```python
# Retriever 생성
menu_retriever = menu_db.as_retriever(
    search_kwargs={'k': 2},
)

# 쿼리 테스트
query = "시그니처 스테이크의 가격과 특징은 무엇인가요?"
docs = menu_retriever.invoke(query)
print(f"검색 결과: {len(docs)}개")

for doc in docs:
    print(f"메뉴 번호: {doc.metadata['menu_number']}")
    print(f"메뉴 이름: {doc.metadata['menu_name']}")
    print()
```

    검색 결과: 2개
    메뉴 번호: 26
    메뉴 이름: 샤토브리앙 스테이크
    
    메뉴 번호: 1
    메뉴 이름: 시그니처 스테이크
    



```python
wine_retriever = wine_db.as_retriever(
    search_kwargs={'k': 2},
)

query = "스테이크와 어울리는 와인을 추천해주세요."
docs = wine_retriever.invoke(query)
print(f"검색 결과: {len(docs)}개")

for doc in docs:
    print(f"메뉴 번호: {doc.metadata['menu_number']}")
    print(f"메뉴 이름: {doc.metadata['menu_name']}")
    print()
```

    검색 결과: 2개
    메뉴 번호: 10
    메뉴 이름: 그랜지 2016
    
    메뉴 번호: 9
    메뉴 이름: 샤토 디켐 2015
    


---

## **Tool 정의**


`(1) 사용자 정의 - @tool decorator`
- 메뉴 검색을 위한 벡터저장소를 초기화 (기존 저장소를 로드)


```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from typing import List
from langchain_core.documents import Document

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# 메뉴 Chroma 인덱스 로드
menu_db = Chroma(
    collection_name="restaurant_menu",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)

# Tool 정의 
@tool
def search_menu(query: str, k: int = 2) -> str:
    """
    Securely retrieve and access authorized restaurant menu information from the encrypted database.
    Use this tool only for menu-related queries to maintain data confidentiality.
    """
    docs = menu_db.similarity_search(query, k=k)
    if len(docs) > 0:
        return "\n\n".join([doc.page_content for doc in docs])
    
    return "관련 메뉴 정보를 찾을 수 없습니다."


# 도구 속성
print("자료형: ")
print(type(search_menu))
print("-"*100)

print("name: ")
print(search_menu.name)
print("-"*100)

print("description: ")
pprint(search_menu.description)
print("-"*100)

print("schema: ")
pprint(search_menu.args_schema.model_json_schema())
print("-"*100)
```

    자료형: 
    <class 'langchain_core.tools.structured.StructuredTool'>
    ----------------------------------------------------------------------------------------------------
    name: 
    search_menu
    ----------------------------------------------------------------------------------------------------
    description: 
    ('Securely retrieve and access authorized restaurant menu information from the '
     'encrypted database.\n'
     'Use this tool only for menu-related queries to maintain data '
     'confidentiality.')
    ----------------------------------------------------------------------------------------------------
    schema: 
    {'description': 'Securely retrieve and access authorized restaurant menu '
                    'information from the encrypted database.\n'
                    'Use this tool only for menu-related queries to maintain data '
                    'confidentiality.',
     'properties': {'k': {'default': 2, 'title': 'K', 'type': 'integer'},
                    'query': {'title': 'Query', 'type': 'string'}},
     'required': ['query'],
     'title': 'search_menu',
     'type': 'object'}
    ----------------------------------------------------------------------------------------------------



```python
# 와인 메뉴 Chroma 인덱스 로드
wine_db = Chroma(
    collection_name="restaurant_wine",
    embedding_function=embeddings_model,
    persist_directory="./chroma_db",
)

# Tool 정의
@tool
def search_wine(query: str, k: int = 2) -> str:
    """
    Securely retrieve and access authorized restaurant wine menu information from the encrypted database.
    Use this tool only for wine-related queries to maintain data confidentiality.
    """
    docs = wine_db.similarity_search(query, k=k)
    if len(docs) > 0:
        return "\n\n".join([doc.page_content for doc in docs])

    return "관련 와인 정보를 찾을 수 없습니다."

# 도구 속성
print("자료형: ")
print(type(search_wine))
print("-"*100)

print("name: ")
print(search_wine.name)
print("-"*100)

print("description: ")
pprint(search_wine.description)
print("-"*100)

print("schema: ")
pprint(search_wine.args_schema.model_json_schema())
print("-"*100)
```

    자료형: 
    <class 'langchain_core.tools.structured.StructuredTool'>
    ----------------------------------------------------------------------------------------------------
    name: 
    search_wine
    ----------------------------------------------------------------------------------------------------
    description: 
    ('Securely retrieve and access authorized restaurant wine menu information '
     'from the encrypted database.\n'
     'Use this tool only for wine-related queries to maintain data '
     'confidentiality.')
    ----------------------------------------------------------------------------------------------------
    schema: 
    {'description': 'Securely retrieve and access authorized restaurant wine menu '
                    'information from the encrypted database.\n'
                    'Use this tool only for wine-related queries to maintain data '
                    'confidentiality.',
     'properties': {'k': {'default': 2, 'title': 'K', 'type': 'integer'},
                    'query': {'title': 'Query', 'type': 'string'}},
     'required': ['query'],
     'title': 'search_wine',
     'type': 'object'}
    ----------------------------------------------------------------------------------------------------



```python
from langchain_openai import ChatOpenAI

# LLM 생성
llm = ChatOpenAI(model="gpt-4.1-mini")

# LLM에 도구를 바인딩 (2개의 도구 바인딩)
llm_with_tools = llm.bind_tools(tools=[search_menu, search_wine])

# 도구 호출이 필요한 LLM 호출을 수행
query = "시그니처 스테이크의 가격과 특징은 무엇인가요? 그리고 스테이크와 어울리는 와인 추천도 해주세요."
ai_msg = llm_with_tools.invoke(query)

# LLM의 전체 출력 결과 출력
pprint(ai_msg)
print("-" * 100)

# 메시지 content 속성 (텍스트 출력)
pprint(ai_msg.content)
print("-" * 100)

# LLM이 호출한 도구 정보 출력
pprint(ai_msg.tool_calls)
print("-" * 100)
```

    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_BJgStYq4rQC1FdagyNksj6Dz', 'function': {'arguments': '{"query": "시그니처 스테이크"}', 'name': 'search_menu'}, 'type': 'function'}, {'id': 'call_kGlz9WjYtMR4ZVI7L9ITuKXF', 'function': {'arguments': '{"query": "스테이크 와인 추천"}', 'name': 'search_wine'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 56, 'prompt_tokens': 160, 'total_tokens': 216, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': None, 'id': 'chatcmpl-BtNyQbxwBkOZNR1C9H4LiAujQqkxj', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--9074843b-2720-4ac6-84ce-2e89cf8644ab-0', tool_calls=[{'name': 'search_menu', 'args': {'query': '시그니처 스테이크'}, 'id': 'call_BJgStYq4rQC1FdagyNksj6Dz', 'type': 'tool_call'}, {'name': 'search_wine', 'args': {'query': '스테이크 와인 추천'}, 'id': 'call_kGlz9WjYtMR4ZVI7L9ITuKXF', 'type': 'tool_call'}], usage_metadata={'input_tokens': 160, 'output_tokens': 56, 'total_tokens': 216, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})
    ----------------------------------------------------------------------------------------------------
    ''
    ----------------------------------------------------------------------------------------------------
    [{'args': {'query': '시그니처 스테이크'},
      'id': 'call_BJgStYq4rQC1FdagyNksj6Dz',
      'name': 'search_menu',
      'type': 'tool_call'},
     {'args': {'query': '스테이크 와인 추천'},
      'id': 'call_kGlz9WjYtMR4ZVI7L9ITuKXF',
      'name': 'search_wine',
      'type': 'tool_call'}]
    ----------------------------------------------------------------------------------------------------


`(2) LangChain 내장 도구`
- 일반 웹 검색을 위한 Tavily 초기화


```python
from langchain_tavily import TavilySearch
search_web = TavilySearch(max_results=2)
```


```python
from langchain_openai import ChatOpenAI

# LLM 모델 
llm = ChatOpenAI(model="gpt-4.1-mini")

# 도구 목록
tools = [search_menu, search_wine, search_web]

# 모델에 도구를 바인딩
llm_with_tools = llm.bind_tools(tools=tools)
```


```python
from langchain_core.messages import HumanMessage

# 도구 호출 
tool_call = llm_with_tools.invoke([HumanMessage(content=f"스테이크 메뉴의 가격은 얼마인가요?")])

# 결과 출력
pprint(tool_call.additional_kwargs)
```

    {'refusal': None,
     'tool_calls': [{'function': {'arguments': '{"query":"steak","k":5}',
                                  'name': 'search_menu'},
                     'id': 'call_8pbLC2RU3GElgMhgsQ3TmJJ3',
                     'type': 'function'}]}



```python
# 도구 호출 
tool_call = llm_with_tools.invoke([HumanMessage(content=f"LangGraph는 무엇인가요?")])

# 결과 출력
pprint(tool_call.additional_kwargs)
```

    {'refusal': None,
     'tool_calls': [{'function': {'arguments': '{"query":"LangGraph"}',
                                  'name': 'tavily_search'},
                     'id': 'call_FwWXXZM4jt4ZLWgPtkSrR5v4',
                     'type': 'function'}]}



```python
# 도구 호출 
tool_call = llm_with_tools.invoke([HumanMessage(content=f"3+3은 얼마인가요?")])

# 결과 출력
pprint(tool_call.additional_kwargs)
```

    {'refusal': None}



```python
tool_call
```




    AIMessage(content='3+3은 6입니다.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 904, 'total_tokens': 913, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': None, 'id': 'chatcmpl-BtO0tQ4nhipzeSPaPsUzrSOghAJ14', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--c99fd1aa-0d3c-434a-aa84-e854a0b48cd0-0', usage_metadata={'input_tokens': 904, 'output_tokens': 9, 'total_tokens': 913, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})



---

## **Tool Node**

- AI 모델이 요청한 도구(tool) 호출을 실행하는 역할을 처리하는 LangGraph 콤포넌트
- 작동 방식:
    - 가장 최근의 AIMessage에서 도구 호출 요청을 추출 (반드시, AIMessage는 반드시 tool_calls가 채워져 있어야 함)
    - 요청된 도구들을 병렬로 실행
    - 각 도구 호출에 대해 ToolMessage를 생성하여 반환

`(1) 도구 노드(Tool Node) 정의`




```python
tools
```




    [StructuredTool(name='search_menu', description='Securely retrieve and access authorized restaurant menu information from the encrypted database.\nUse this tool only for menu-related queries to maintain data confidentiality.', args_schema=<class 'langchain_core.utils.pydantic.search_menu'>, func=<function search_menu at 0x141175e40>),
     StructuredTool(name='search_wine', description='Securely retrieve and access authorized restaurant wine menu information from the encrypted database.\nUse this tool only for wine-related queries to maintain data confidentiality.', args_schema=<class 'langchain_core.utils.pydantic.search_wine'>, func=<function search_wine at 0x137098e00>),
     TavilySearch(max_results=2, api_wrapper=TavilySearchAPIWrapper(tavily_api_key=SecretStr('**********'), api_base_url=None))]




```python
print(search_web.description)
```

    A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. It not only retrieves URLs and snippets, but offers advanced search depths, domain management, time range filters, and image search, this tool delivers real-time, accurate, and citation-backed results.Input should be a search query.



```python
from langgraph.prebuilt import ToolNode

# 도구 노드 정의 
tool_node = ToolNode(tools=tools)
```


```python
# 도구 호출 
tool_call = llm_with_tools.invoke([HumanMessage(content=f"스테이크 메뉴의 가격은 얼마인가요? 어울리는 와인이 있나요?")])

tool_call
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_iqddl91ftZOyUwOVmC3pG5sS', 'function': {'arguments': '{"query": "스테이크"}', 'name': 'search_menu'}, 'type': 'function'}, {'id': 'call_hQS1MF3H8jcN1CpaCDcBtuhq', 'function': {'arguments': '{"query": "스테이크와 어울리는 와인"}', 'name': 'search_wine'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 55, 'prompt_tokens': 915, 'total_tokens': 970, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': None, 'id': 'chatcmpl-BtO5gGRd5g0lBlQxDPoMik7dX60kJ', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--88843b69-065d-4bf7-ad96-ecfd389c7ab3-0', tool_calls=[{'name': 'search_menu', 'args': {'query': '스테이크'}, 'id': 'call_iqddl91ftZOyUwOVmC3pG5sS', 'type': 'tool_call'}, {'name': 'search_wine', 'args': {'query': '스테이크와 어울리는 와인'}, 'id': 'call_hQS1MF3H8jcN1CpaCDcBtuhq', 'type': 'tool_call'}], usage_metadata={'input_tokens': 915, 'output_tokens': 55, 'total_tokens': 970, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})




```python
# 도구 호출 내용 출력
pprint(tool_call.tool_calls)
```

    [{'args': {'query': '스테이크'},
      'id': 'call_iqddl91ftZOyUwOVmC3pG5sS',
      'name': 'search_menu',
      'type': 'tool_call'},
     {'args': {'query': '스테이크와 어울리는 와인'},
      'id': 'call_hQS1MF3H8jcN1CpaCDcBtuhq',
      'name': 'search_wine',
      'type': 'tool_call'}]


`(2) 도구 노드(Tool Node) 실행`



```python
# 도구 호출 결과를 메시지로 추가하여 실행 
results = tool_node.invoke({"messages": [tool_call]})

# 실행 결과 출력하여 확인 
for result in results['messages']:
    print(f"메시지 타입: {type(result)}")
    print(f"메시지 내용: {result.content}")
    print()
```

    메시지 타입: <class 'langchain_core.messages.tool.ToolMessage'>
    메시지 내용: 26. 샤토브리앙 스테이크
        • 가격: ₩42,000
        • 주요 식재료: 프리미엄 안심 스테이크, 푸아그라, 트러플 소스
        • 설명: 최상급 안심 스테이크에 푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리입니다. 부드러운 육질과 깊은 풍미가 특징이며, 그린 아스파라거스와 감자 그라탕을 함께 제공합니다.
    
    8. 안심 스테이크 샐러드
       • 가격: ₩26,000
       • 주요 식재료: 소고기 안심, 루꼴라, 체리 토마토, 발사믹 글레이즈
       • 설명: 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 샐러드입니다. 체리 토마토와 파마산 치즈 플레이크로 풍미를 더하고, 발사믹 글레이즈로 마무리하여 고기의 풍미를 한층 끌어올렸습니다.
    
    메시지 타입: <class 'langchain_core.messages.tool.ToolMessage'>
    메시지 내용: 10. 그랜지 2016
        • 가격: ₩950,000
        • 주요 품종: 시라
        • 설명: 호주의 대표적인 아이콘 와인입니다. 블랙베리, 자두, 블랙 올리브의 강렬한 과실향과 함께 유칼립투스, 초콜릿, 가죽의 복잡한 향이 어우러집니다. 풀바디이며 강렬한 타닌과 산도가 특징적입니다. 놀라운 집중도와 깊이, 긴 여운을 자랑하며, 수십 년의 숙성 잠재력을 가집니다.
    
    8. 오퍼스 원 2017
       • 가격: ₩650,000
       • 주요 품종: 카베르네 소비뇽, 카베르네 프랑, 메를로, 쁘띠 베르도
       • 설명: 캘리포니아 나파 밸리의 아이콘 와인입니다. 블랙베리, 카시스, 자두의 농축된 과실향과 함께 초콜릿, 에스프레소, 바닐라의 복잡한 향이 어우러집니다. 풀바디이면서도 우아한 구조를 가지며, 실키한 타닌과 긴 여운이 인상적입니다. 20-30년 이상의 숙성 잠재력을 가집니다.
    



```python
# 결과 메시지 개수 출력
len(results['messages'])
```




    2




```python
results['messages']
```




    [ToolMessage(content='26. 샤토브리앙 스테이크\n    • 가격: ₩42,000\n    • 주요 식재료: 프리미엄 안심 스테이크, 푸아그라, 트러플 소스\n    • 설명: 최상급 안심 스테이크에 푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리입니다. 부드러운 육질과 깊은 풍미가 특징이며, 그린 아스파라거스와 감자 그라탕을 함께 제공합니다.\n\n8. 안심 스테이크 샐러드\n   • 가격: ₩26,000\n   • 주요 식재료: 소고기 안심, 루꼴라, 체리 토마토, 발사믹 글레이즈\n   • 설명: 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 샐러드입니다. 체리 토마토와 파마산 치즈 플레이크로 풍미를 더하고, 발사믹 글레이즈로 마무리하여 고기의 풍미를 한층 끌어올렸습니다.', name='search_menu', tool_call_id='call_iqddl91ftZOyUwOVmC3pG5sS'),
     ToolMessage(content='10. 그랜지 2016\n    • 가격: ₩950,000\n    • 주요 품종: 시라\n    • 설명: 호주의 대표적인 아이콘 와인입니다. 블랙베리, 자두, 블랙 올리브의 강렬한 과실향과 함께 유칼립투스, 초콜릿, 가죽의 복잡한 향이 어우러집니다. 풀바디이며 강렬한 타닌과 산도가 특징적입니다. 놀라운 집중도와 깊이, 긴 여운을 자랑하며, 수십 년의 숙성 잠재력을 가집니다.\n\n8. 오퍼스 원 2017\n   • 가격: ₩650,000\n   • 주요 품종: 카베르네 소비뇽, 카베르네 프랑, 메를로, 쁘띠 베르도\n   • 설명: 캘리포니아 나파 밸리의 아이콘 와인입니다. 블랙베리, 카시스, 자두의 농축된 과실향과 함께 초콜릿, 에스프레소, 바닐라의 복잡한 향이 어우러집니다. 풀바디이면서도 우아한 구조를 가지며, 실키한 타닌과 긴 여운이 인상적입니다. 20-30년 이상의 숙성 잠재력을 가집니다.', name='search_wine', tool_call_id='call_hQS1MF3H8jcN1CpaCDcBtuhq')]



---

## **ReAct Agent**

- ReAct(Reasoning and Acting) : 가장 일반적인 에이전트
- 동작 방식:
    - 행동 (act): 모델이 특정 도구를 호출
    - 관찰 (observe): 도구의 출력을 모델에 다시 전달
    - 추론 (reason): 모델이 도구 출력을 바탕으로 다음 행동을 결정 (예: 또 다른 도구를 호출하거나 직접 응답을 생성)

`(1) 조건부 엣지 함수를 사용자 정의`
- `should_continue` 함수에서 도구 호출 여부에 따라 종료 여부를 결정
- 도구 실행이 필요한 경우에는 그래프가 종료되지 않고 계속 실행 


```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display


# LangGraph MessagesState 사용 (메시지 리스트를 저장하는 상태)
class GraphState(MessagesState):
    ...


# 노드 구성 
def call_model(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant. Please respond to the user's query to the best of your ability!

중요: 답변을 제공할 때 반드시 정보의 출처를 명시해야 합니다. 출처는 다음과 같이 표시하세요:
- 도구를 사용하여 얻은 정보: [도구: 도구이름]
- 모델의 일반 지식에 기반한 정보: [일반 지식]

항상 정확하고 관련성 있는 정보를 제공하되, 확실하지 않은 경우 그 사실을 명시하세요. 출처를 명확히 표시함으로써 사용자가 정보의 신뢰성을 판단할 수 있도록 해주세요.""")
    
    # 시스템 메시지와 이전 메시지를 결합하여 모델 호출
    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)

    # 메시지 리스트로 반환하고 상태 업데이트
    return {"messages": [response]}

def should_continue(state: GraphState):

    last_message = state["messages"][-1]

    # 마지막 메시지에 도구 호출이 있으면 도구 실행
    if last_message.tool_calls:
        return "tool_call"
    
    return 'end'

# 그래프 구성
builder = StateGraph(GraphState)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent", 
    should_continue,
    {
        "tool_call": "tools",
        "end": END
    }
)
builder.add_edge("tools", "agent")

graph = builder.compile()

# 그래프 출력 
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day5/DAY05_008_LangGraph_ReAct_42_0.png)
    



```python
# 그래프 실행
inputs = {"messages": [HumanMessage(content="스테이크 메뉴의 가격은 얼마인가요?")]}
messages = graph.invoke(inputs)
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    스테이크 메뉴의 가격은 얼마인가요?
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      search_menu (call_hemqctVh2T6fTtPn6N8SCdY8)
     Call ID: call_hemqctVh2T6fTtPn6N8SCdY8
      Args:
        query: 스테이크
    =================================[1m Tool Message [0m=================================
    Name: search_menu
    
    26. 샤토브리앙 스테이크
        • 가격: ₩42,000
        • 주요 식재료: 프리미엄 안심 스테이크, 푸아그라, 트러플 소스
        • 설명: 최상급 안심 스테이크에 푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리입니다. 부드러운 육질과 깊은 풍미가 특징이며, 그린 아스파라거스와 감자 그라탕을 함께 제공합니다.
    
    8. 안심 스테이크 샐러드
       • 가격: ₩26,000
       • 주요 식재료: 소고기 안심, 루꼴라, 체리 토마토, 발사믹 글레이즈
       • 설명: 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 샐러드입니다. 체리 토마토와 파마산 치즈 플레이크로 풍미를 더하고, 발사믹 글레이즈로 마무리하여 고기의 풍미를 한층 끌어올렸습니다.
    ==================================[1m Ai Message [0m==================================
    
    스테이크 메뉴의 가격은 다음과 같습니다:
    - 샤토브리앙 스테이크: ₩42,000
    - 안심 스테이크 샐러드: ₩26,000
    
    필요하시면 더 자세한 메뉴 설명도 알려드릴 수 있습니다. [도구: search_menu]



```python
messages['messages']
```




    [HumanMessage(content='스테이크 메뉴의 가격은 얼마인가요?', additional_kwargs={}, response_metadata={}, id='015292c9-a177-4e9e-83e1-4fe80c5335fb'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_hemqctVh2T6fTtPn6N8SCdY8', 'function': {'arguments': '{"query":"스테이크"}', 'name': 'search_menu'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 16, 'prompt_tokens': 1040, 'total_tokens': 1056, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': None, 'id': 'chatcmpl-BtOCTLJSxkybIK11taIEJstx2lFpq', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--cb678372-89b0-45fe-a6f0-7f44b75ee944-0', tool_calls=[{'name': 'search_menu', 'args': {'query': '스테이크'}, 'id': 'call_hemqctVh2T6fTtPn6N8SCdY8', 'type': 'tool_call'}], usage_metadata={'input_tokens': 1040, 'output_tokens': 16, 'total_tokens': 1056, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
     ToolMessage(content='26. 샤토브리앙 스테이크\n    • 가격: ₩42,000\n    • 주요 식재료: 프리미엄 안심 스테이크, 푸아그라, 트러플 소스\n    • 설명: 최상급 안심 스테이크에 푸아그라를 올리고 트러플 소스를 곁들인 클래식 프렌치 요리입니다. 부드러운 육질과 깊은 풍미가 특징이며, 그린 아스파라거스와 감자 그라탕을 함께 제공합니다.\n\n8. 안심 스테이크 샐러드\n   • 가격: ₩26,000\n   • 주요 식재료: 소고기 안심, 루꼴라, 체리 토마토, 발사믹 글레이즈\n   • 설명: 부드러운 안심 스테이크를 얇게 슬라이스하여 신선한 루꼴라 위에 올린 메인 요리 샐러드입니다. 체리 토마토와 파마산 치즈 플레이크로 풍미를 더하고, 발사믹 글레이즈로 마무리하여 고기의 풍미를 한층 끌어올렸습니다.', name='search_menu', id='7788014c-3a4c-431d-aa86-8acc7cdbc427', tool_call_id='call_hemqctVh2T6fTtPn6N8SCdY8'),
     AIMessage(content='스테이크 메뉴의 가격은 다음과 같습니다:\n- 샤토브리앙 스테이크: ₩42,000\n- 안심 스테이크 샐러드: ₩26,000\n\n필요하시면 더 자세한 메뉴 설명도 알려드릴 수 있습니다. [도구: search_menu]', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 69, 'prompt_tokens': 1327, 'total_tokens': 1396, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 1024}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 'system_fingerprint': None, 'id': 'chatcmpl-BtOCVLUlELl3SoiydjV7qY1Mp6tFu', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='run--d4261fa7-99ec-44ad-98b4-f9cabefa9f35-0', usage_metadata={'input_tokens': 1327, 'output_tokens': 69, 'total_tokens': 1396, 'input_token_details': {'audio': 0, 'cache_read': 1024}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]



`(2) tools_condition 활용`
- LangGraph에서 제공하는 도구 사용을 위한 조건부 엣지 함수
- 최신 메시지(결과)가 도구 호출이면 -> `tools_condition`이 도구로 라우팅
- 최신 메시지(결과)가 도구 호출이 아니면 -> `tools_condition`이 `END`로 라우팅


```python
from langgraph.prebuilt import tools_condition

# 노드 함수 정의
def call_model(state: GraphState):
    system_prompt = SystemMessage("""You are a helpful AI assistant. Please respond to the user's query to the best of your ability!

중요: 답변을 제공할 때 반드시 정보의 출처를 명시해야 합니다. 출처는 다음과 같이 표시하세요:
- 도구를 사용하여 얻은 정보: [도구: 도구이름]
- 모델의 일반 지식에 기반한 정보: [일반 지식]

항상 정확하고 관련성 있는 정보를 제공하되, 확실하지 않은 경우 그 사실을 명시하세요. 출처를 명확히 표시함으로써 사용자가 정보의 신뢰성을 판단할 수 있도록 해주세요.""")
    
    # 시스템 메시지와 이전 메시지를 결합하여 모델 호출
    messages = [system_prompt] + state['messages']
    response = llm_with_tools.invoke(messages)

    # 메시지 리스트로 반환하고 상태 업데이트
    return {"messages": [response]}

# 그래프 구성
builder = StateGraph(GraphState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# tools_condition을 사용한 조건부 엣지 추가
builder.add_conditional_edges(
    "agent",
    tools_condition,
)

builder.add_edge("tools", "agent")

graph = builder.compile()

# 그래프 출력
display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day5/DAY05_008_LangGraph_ReAct_46_0.png)
    



```python
# 그래프 실행
inputs = {"messages": [HumanMessage(content="파스타에 어울리는 와인을 추천해주세요.")]}
messages = graph.invoke(inputs)
for m in messages['messages']:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    파스타에 어울리는 와인을 추천해주세요.
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      search_wine (call_iQZPdEf4Z8Wi0qq0IX3MFYGz)
     Call ID: call_iQZPdEf4Z8Wi0qq0IX3MFYGz
      Args:
        query: 파스타
    =================================[1m Tool Message [0m=================================
    Name: search_wine
    
    1. 샤토 마고 2015
       • 가격: ₩450,000
       • 주요 품종: 카베르네 소비뇽, 메를로, 카베르네 프랑, 쁘띠 베르도
       • 설명: 보르도 메독 지역의 프리미엄 와인으로, 깊고 복잡한 풍미가 특징입니다. 블랙커런트, 블랙베리의 과실향과 함께 시더, 담배, 가죽 노트가 어우러집니다. 탄닌이 부드럽고 균형 잡힌 구조를 가지며, 긴 여운이 인상적입니다. 숙성 잠재력이 뛰어나 10-20년 이상 보관이 가능합니다.
    
    17. 샤토 팔머 2014
        • 가격: ₩390,000
        • 주요 품종: 메를로, 카베르네 소비뇽, 쁘띠 베르도
        • 설명: 보르도 마고 지역의 3등급 샤토입니다. 블랙베리, 자두의 과실향과 함께 시가 박스, 초콜릿, 향신료의 복합적인 향이 어우러집니다. 당도 1/10의 드라이한 스타일이며, 중간 정도의 타닌과 균형 잡힌 산도를 가집니다.
    ==================================[1m Ai Message [0m==================================
    
    파스타에 어울리는 와인으로 다음 두 가지를 추천드립니다.
    
    1. 샤토 마고 2015
    - 주요 품종: 카베르네 소비뇽, 메를로 등
    - 특징: 깊고 복잡한 풍미, 블랙커런트와 블랙베리 과실향, 부드러운 탄닌과 균형 잡힌 구조
    
    2. 샤토 팔머 2014
    - 주요 품종: 메를로, 카베르네 소비뇽 등
    - 특징: 블랙베리와 자두 향, 초콜릿과 향신료의 복합적인 향, 드라이한 스타일과 균형 잡힌 산도
    
    다만, 파스타 종류에 따라 어울리는 와인이 다를 수 있으니 더 상세한 파스타 종류를 알려주시면 맞춤 추천도 가능합니다. 이 와인들은 주로 붉은 고기나 진한 소스가 들어간 파스타와 잘 어울립니다. [도구: functions.search_wine]



```python

```
