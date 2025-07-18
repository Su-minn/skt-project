#   LangGraph 활용 - Multi-Agent 아키텍처

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

import warnings
warnings.filterwarnings("ignore")
```

`(3) Langsmith tracing 설정`


```python
# Langsmith tracing 여부를 확인 (true: langsmith 추척 활성화, false: langsmith 추척 비활성화)
import os
print(os.getenv('LANGSMITH_TRACING'))
```

    true


---

## **LangGraph 멀티에이전트**


**1. 에이전트란?**
  - **에이전트**는 LLM을 사용하여 애플리케이션의 제어 흐름을 결정하는 시스템

**2. 멀티 에이전트 시스템이 필요한 이유**
  - 단일 에이전트 시스템이 복잡해지면서 다음과 같은 문제가 발생:
    - **도구 과부하**: 에이전트가 너무 많은 도구를 가져 잘못된 결정을 내림
    - **컨텍스트 복잡성**: 단일 에이전트가 추적하기에 너무 복잡한 컨텍스트
    - **전문화 필요**: 플래너, 연구자, 수학 전문가 등 여러 전문 영역이 필요

**3. 멀티 에이전트 시스템의 주요 장점**
  - **모듈성**: 개별 에이전트로 분리하여 개발, 테스트, 유지보수가 용이
  - **전문화**: 특정 도메인에 초점을 맞춘 전문 에이전트 생성으로 전체 성능 향상
  - **제어**: 에이전트 간 통신을 명시적으로 제어 가능

### **1. Supervisor 패턴** 

- **특징**: 단일 슈퍼바이저 에이전트가 다른 에이전트들의 실행을 결정
- **적용**: 중앙 집중식 제어가 필요한 경우

![Supervisor 패턴](https://langchain-ai.github.io/langgraph/agents/assets/supervisor.png)

`(1) langgraph-supervisor 패키지 사용`

- **설치**

    ```bash
    pip install langgraph-supervisor 
    ```

    ```bash
    uv add langgraph-supervisor
    ```


```python
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from IPython.display import Image, display

# 1. 작업자 에이전트들 생성
# 연구 에이전트

tavily_search = TavilySearch(max_results=3)

research_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[tavily_search],
    prompt="당신은 연구 전문가입니다. 웹 검색으로 정보를 찾아주세요.",
    name="research_agent"
)

# 수학 에이전트
@tool
def add(a: float, b: float) -> float:
    """두 수를 더합니다."""
    return a + b

@tool  
def multiply(a: float, b: float) -> float:
    """두 수를 곱합니다."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """두 수를 나눕니다."""
    if b == 0:
        return "0으로 나눌 수 없습니다."
    return a / b

math_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[add, multiply, divide],
    prompt="당신은 수학 전문가입니다. 계산을 도와드립니다.",
    name="math_agent"
)

# 2. 감독자 시스템 생성 (간단한 방법: langgraph-supervisor 패키지 사용)
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),
    agents=[research_agent, math_agent],
    prompt="""
    당신은 감독자입니다. 두 명의 에이전트를 관리합니다:
    - research_agent: 정보 검색 작업을 전담 
    - math_agent: 수학 계산 작업을 전담

    전문성을 고려하여 적절한 에이전트에게 작업을 할당하세요.
    """,
).compile()
    

# 그래프 시각화
display(Image(supervisor.get_graph(xray=False).draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_004_LangGraph_Multi-Agent_11_0.png)
    



```python
# 3. 실행
result = supervisor.invoke({
    "messages": [{
        "role": "user", 
        "content": "한국의 인구를 찾아서, 인구 수에 2를 곱해주세요."
    }]
})

for m in result["messages"]:
    m.pretty_print()
```

    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel is_last_step, ignoring it.
    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel remaining_steps, ignoring it.
    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel is_last_step, ignoring it.
    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel remaining_steps, ignoring it.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[21], line 2
          1 # 3. 실행
    ----> 2 result = supervisor.invoke({
          3     "messages": [{
          4         "role": "user", 
          5         "content": "한국의 인구를 찾아서, 인구 수에 2를 곱해주세요."
          6     }]
          7 })
          9 for m in result["messages"]:
         10     m.pretty_print()


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py:2844, in Pregel.invoke(self, input, config, stream_mode, print_mode, output_keys, interrupt_before, interrupt_after, **kwargs)
       2841 chunks: list[dict[str, Any] | Any] = []
       2842 interrupts: list[Interrupt] = []
    -> 2844 for chunk in self.stream(
       2845     input,
       2846     config,
       2847     stream_mode=["updates", "values"]
       2848     if stream_mode == "values"
       2849     else stream_mode,
       2850     print_mode=print_mode,
       2851     output_keys=output_keys,
       2852     interrupt_before=interrupt_before,
       2853     interrupt_after=interrupt_after,
       2854     **kwargs,
       2855 ):
       2856     if stream_mode == "values":
       2857         if len(chunk) == 2:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py:2534, in Pregel.stream(self, input, config, stream_mode, print_mode, output_keys, interrupt_before, interrupt_after, checkpoint_during, debug, subgraphs)
       2532 for task in loop.match_cached_writes():
       2533     loop.output_writes(task.id, task.writes, cached=True)
    -> 2534 for _ in runner.tick(
       2535     [t for t in loop.tasks.values() if not t.writes],
       2536     timeout=self.step_timeout,
       2537     get_waiter=get_waiter,
       2538     schedule_task=loop.accept_push,
       2539 ):
       2540     # emit output
       2541     yield from _output(
       2542         stream_mode, print_mode, subgraphs, stream.get, queue.Empty
       2543     )
       2544 loop.after_tick()


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/runner.py:162, in PregelRunner.tick(self, tasks, reraise, timeout, retry_policy, get_waiter, schedule_task)
        160 t = tasks[0]
        161 try:
    --> 162     run_with_retry(
        163         t,
        164         retry_policy,
        165         configurable={
        166             CONFIG_KEY_CALL: partial(
        167                 _call,
        168                 weakref.ref(t),
        169                 retry_policy=retry_policy,
        170                 futures=weakref.ref(futures),
        171                 schedule_task=schedule_task,
        172                 submit=self.submit,
        173             ),
        174         },
        175     )
        176     self.commit(t, None)
        177 except Exception as exc:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/retry.py:42, in run_with_retry(task, retry_policy, configurable)
         40     task.writes.clear()
         41     # run the task
    ---> 42     return task.proc.invoke(task.input, config)
         43 except ParentCommand as exc:
         44     ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/utils/runnable.py:623, in RunnableSeq.invoke(self, input, config, **kwargs)
        621     # run in context
        622     with set_config_context(config, run) as context:
    --> 623         input = context.run(step.invoke, input, config, **kwargs)
        624 else:
        625     input = step.invoke(input, config)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py:2844, in Pregel.invoke(self, input, config, stream_mode, print_mode, output_keys, interrupt_before, interrupt_after, **kwargs)
       2841 chunks: list[dict[str, Any] | Any] = []
       2842 interrupts: list[Interrupt] = []
    -> 2844 for chunk in self.stream(
       2845     input,
       2846     config,
       2847     stream_mode=["updates", "values"]
       2848     if stream_mode == "values"
       2849     else stream_mode,
       2850     print_mode=print_mode,
       2851     output_keys=output_keys,
       2852     interrupt_before=interrupt_before,
       2853     interrupt_after=interrupt_after,
       2854     **kwargs,
       2855 ):
       2856     if stream_mode == "values":
       2857         if len(chunk) == 2:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/__init__.py:2534, in Pregel.stream(self, input, config, stream_mode, print_mode, output_keys, interrupt_before, interrupt_after, checkpoint_during, debug, subgraphs)
       2532 for task in loop.match_cached_writes():
       2533     loop.output_writes(task.id, task.writes, cached=True)
    -> 2534 for _ in runner.tick(
       2535     [t for t in loop.tasks.values() if not t.writes],
       2536     timeout=self.step_timeout,
       2537     get_waiter=get_waiter,
       2538     schedule_task=loop.accept_push,
       2539 ):
       2540     # emit output
       2541     yield from _output(
       2542         stream_mode, print_mode, subgraphs, stream.get, queue.Empty
       2543     )
       2544 loop.after_tick()


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/runner.py:162, in PregelRunner.tick(self, tasks, reraise, timeout, retry_policy, get_waiter, schedule_task)
        160 t = tasks[0]
        161 try:
    --> 162     run_with_retry(
        163         t,
        164         retry_policy,
        165         configurable={
        166             CONFIG_KEY_CALL: partial(
        167                 _call,
        168                 weakref.ref(t),
        169                 retry_policy=retry_policy,
        170                 futures=weakref.ref(futures),
        171                 schedule_task=schedule_task,
        172                 submit=self.submit,
        173             ),
        174         },
        175     )
        176     self.commit(t, None)
        177 except Exception as exc:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/pregel/retry.py:42, in run_with_retry(task, retry_policy, configurable)
         40     task.writes.clear()
         41     # run the task
    ---> 42     return task.proc.invoke(task.input, config)
         43 except ParentCommand as exc:
         44     ns: str = config[CONF][CONFIG_KEY_CHECKPOINT_NS]


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/utils/runnable.py:623, in RunnableSeq.invoke(self, input, config, **kwargs)
        621     # run in context
        622     with set_config_context(config, run) as context:
    --> 623         input = context.run(step.invoke, input, config, **kwargs)
        624 else:
        625     input = step.invoke(input, config)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/utils/runnable.py:370, in RunnableCallable.invoke(self, input, config, **kwargs)
        368     # run in context
        369     with set_config_context(child_config, run) as context:
    --> 370         ret = context.run(self.func, *args, **kwargs)
        371 except BaseException as e:
        372     run_manager.on_chain_error(e)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py:507, in create_react_agent.<locals>.call_model(state, config)
        505 def call_model(state: StateSchema, config: RunnableConfig) -> StateSchema:
        506     state = _get_model_input_state(state)
    --> 507     response = cast(AIMessage, model_runnable.invoke(state, config))
        508     # add agent name to the AIMessage
        509     response.name = name


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_core/runnables/base.py:3047, in RunnableSequence.invoke(self, input, config, **kwargs)
       3045                 input_ = context.run(step.invoke, input_, config, **kwargs)
       3046             else:
    -> 3047                 input_ = context.run(step.invoke, input_, config)
       3048 # finish the root run
       3049 except BaseException as e:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_core/runnables/base.py:5431, in RunnableBindingBase.invoke(self, input, config, **kwargs)
       5424 @override
       5425 def invoke(
       5426     self,
       (...)   5429     **kwargs: Optional[Any],
       5430 ) -> Output:
    -> 5431     return self.bound.invoke(
       5432         input,
       5433         self._merge_configs(config),
       5434         **{**self.kwargs, **kwargs},
       5435     )


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:1326, in ChatGoogleGenerativeAI.invoke(self, input, config, code_execution, stop, **kwargs)
       1321     else:
       1322         raise ValueError(
       1323             "Tools are already defined." "code_execution tool can't be defined"
       1324         )
    -> 1326 return super().invoke(input, config, stop=stop, **kwargs)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_core/language_models/chat_models.py:378, in BaseChatModel.invoke(self, input, config, stop, **kwargs)
        366 @override
        367 def invoke(
        368     self,
       (...)    373     **kwargs: Any,
        374 ) -> BaseMessage:
        375     config = ensure_config(config)
        376     return cast(
        377         "ChatGeneration",
    --> 378         self.generate_prompt(
        379             [self._convert_input(input)],
        380             stop=stop,
        381             callbacks=config.get("callbacks"),
        382             tags=config.get("tags"),
        383             metadata=config.get("metadata"),
        384             run_name=config.get("run_name"),
        385             run_id=config.pop("run_id", None),
        386             **kwargs,
        387         ).generations[0][0],
        388     ).message


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_core/language_models/chat_models.py:963, in BaseChatModel.generate_prompt(self, prompts, stop, callbacks, **kwargs)
        954 @override
        955 def generate_prompt(
        956     self,
       (...)    960     **kwargs: Any,
        961 ) -> LLMResult:
        962     prompt_messages = [p.to_messages() for p in prompts]
    --> 963     return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_core/language_models/chat_models.py:782, in BaseChatModel.generate(self, messages, stop, callbacks, tags, metadata, run_name, run_id, **kwargs)
        779 for i, m in enumerate(input_messages):
        780     try:
        781         results.append(
    --> 782             self._generate_with_cache(
        783                 m,
        784                 stop=stop,
        785                 run_manager=run_managers[i] if run_managers else None,
        786                 **kwargs,
        787             )
        788         )
        789     except BaseException as e:
        790         if run_managers:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_core/language_models/chat_models.py:1028, in BaseChatModel._generate_with_cache(self, messages, stop, run_manager, **kwargs)
       1026     result = generate_from_stream(iter(chunks))
       1027 elif inspect.signature(self._generate).parameters.get("run_manager"):
    -> 1028     result = self._generate(
       1029         messages, stop=stop, run_manager=run_manager, **kwargs
       1030     )
       1031 else:
       1032     result = self._generate(messages, stop=stop, **kwargs)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:1433, in ChatGoogleGenerativeAI._generate(self, messages, stop, run_manager, tools, functions, safety_settings, tool_config, generation_config, cached_content, tool_choice, **kwargs)
       1406 def _generate(
       1407     self,
       1408     messages: List[BaseMessage],
       (...)   1419     **kwargs: Any,
       1420 ) -> ChatResult:
       1421     request = self._prepare_request(
       1422         messages,
       1423         stop=stop,
       (...)   1431         **kwargs,
       1432     )
    -> 1433     response: GenerateContentResponse = _chat_with_retry(
       1434         request=request,
       1435         **kwargs,
       1436         generation_method=self.client.generate_content,
       1437         metadata=self.default_metadata,
       1438     )
       1439     return _response_to_result(response)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:231, in _chat_with_retry(generation_method, **kwargs)
        222         raise e
        224 params = (
        225     {k: v for k, v in kwargs.items() if k in _allowed_params_prediction_service}
        226     if (request := kwargs.get("request"))
       (...)    229     else kwargs
        230 )
    --> 231 return _chat_with_retry(**params)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/tenacity/__init__.py:338, in BaseRetrying.wraps.<locals>.wrapped_f(*args, **kw)
        336 copy = self.copy()
        337 wrapped_f.statistics = copy.statistics  # type: ignore[attr-defined]
    --> 338 return copy(f, *args, **kw)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/tenacity/__init__.py:477, in Retrying.__call__(self, fn, *args, **kwargs)
        475 retry_state = RetryCallState(retry_object=self, fn=fn, args=args, kwargs=kwargs)
        476 while True:
    --> 477     do = self.iter(retry_state=retry_state)
        478     if isinstance(do, DoAttempt):
        479         try:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/tenacity/__init__.py:378, in BaseRetrying.iter(self, retry_state)
        376 result = None
        377 for action in self.iter_state.actions:
    --> 378     result = action(retry_state)
        379 return result


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/tenacity/__init__.py:400, in BaseRetrying._post_retry_check_actions.<locals>.<lambda>(rs)
        398 def _post_retry_check_actions(self, retry_state: "RetryCallState") -> None:
        399     if not (self.iter_state.is_explicit_retry or self.iter_state.retry_run_result):
    --> 400         self._add_action_func(lambda rs: rs.outcome.result())
        401         return
        403     if self.after is not None:


    File ~/miniconda3/lib/python3.12/concurrent/futures/_base.py:449, in Future.result(self, timeout)
        447     raise CancelledError()
        448 elif self._state == FINISHED:
    --> 449     return self.__get_result()
        451 self._condition.wait(timeout)
        453 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:


    File ~/miniconda3/lib/python3.12/concurrent/futures/_base.py:401, in Future.__get_result(self)
        399 if self._exception:
        400     try:
    --> 401         raise self._exception
        402     finally:
        403         # Break a reference cycle with the exception in self._exception
        404         self = None


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/tenacity/__init__.py:480, in Retrying.__call__(self, fn, *args, **kwargs)
        478 if isinstance(do, DoAttempt):
        479     try:
    --> 480         result = fn(*args, **kwargs)
        481     except BaseException:  # noqa: B902
        482         retry_state.set_exception(sys.exc_info())  # type: ignore[arg-type]


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/langchain_google_genai/chat_models.py:206, in _chat_with_retry.<locals>._chat_with_retry(**kwargs)
        203 @retry_decorator
        204 def _chat_with_retry(**kwargs: Any) -> Any:
        205     try:
    --> 206         return generation_method(**kwargs)
        207     # Do not retry for these errors.
        208     except google.api_core.exceptions.FailedPrecondition as exc:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/ai/generativelanguage_v1beta/services/generative_service/client.py:868, in GenerativeServiceClient.generate_content(self, request, model, contents, retry, timeout, metadata)
        865 self._validate_universe_domain()
        867 # Send the request.
    --> 868 response = rpc(
        869     request,
        870     retry=retry,
        871     timeout=timeout,
        872     metadata=metadata,
        873 )
        875 # Done; return the response.
        876 return response


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/api_core/gapic_v1/method.py:131, in _GapicCallable.__call__(self, timeout, retry, compression, *args, **kwargs)
        128 if self._compression is not None:
        129     kwargs["compression"] = compression
    --> 131 return wrapped_func(*args, **kwargs)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/api_core/retry/retry_unary.py:294, in Retry.__call__.<locals>.retry_wrapped_func(*args, **kwargs)
        290 target = functools.partial(func, *args, **kwargs)
        291 sleep_generator = exponential_sleep_generator(
        292     self._initial, self._maximum, multiplier=self._multiplier
        293 )
    --> 294 return retry_target(
        295     target,
        296     self._predicate,
        297     sleep_generator,
        298     timeout=self._timeout,
        299     on_error=on_error,
        300 )


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/api_core/retry/retry_unary.py:147, in retry_target(target, predicate, sleep_generator, timeout, on_error, exception_factory, **kwargs)
        145 while True:
        146     try:
    --> 147         result = target()
        148         if inspect.isawaitable(result):
        149             warnings.warn(_ASYNC_RETRY_WARNING)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/api_core/timeout.py:130, in TimeToDeadlineTimeout.__call__.<locals>.func_with_timeout(*args, **kwargs)
        126         remaining_timeout = self._timeout
        128     kwargs["timeout"] = remaining_timeout
    --> 130 return func(*args, **kwargs)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/api_core/grpc_helpers.py:76, in _wrap_unary_errors.<locals>.error_remapped_callable(*args, **kwargs)
         73 @functools.wraps(callable_)
         74 def error_remapped_callable(*args, **kwargs):
         75     try:
    ---> 76         return callable_(*args, **kwargs)
         77     except grpc.RpcError as exc:
         78         raise exceptions.from_grpc_error(exc) from exc


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/grpc/_interceptor.py:277, in _UnaryUnaryMultiCallable.__call__(self, request, timeout, metadata, credentials, wait_for_ready, compression)
        268 def __call__(
        269     self,
        270     request: Any,
       (...)    275     compression: Optional[grpc.Compression] = None,
        276 ) -> Any:
    --> 277     response, ignored_call = self._with_call(
        278         request,
        279         timeout=timeout,
        280         metadata=metadata,
        281         credentials=credentials,
        282         wait_for_ready=wait_for_ready,
        283         compression=compression,
        284     )
        285     return response


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/grpc/_interceptor.py:329, in _UnaryUnaryMultiCallable._with_call(self, request, timeout, metadata, credentials, wait_for_ready, compression)
        326     except Exception as exception:  # pylint:disable=broad-except
        327         return _FailureOutcome(exception, sys.exc_info()[2])
    --> 329 call = self._interceptor.intercept_unary_unary(
        330     continuation, client_call_details, request
        331 )
        332 return call.result(), call


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/google/ai/generativelanguage_v1beta/services/generative_service/transports/grpc.py:78, in _LoggingClientInterceptor.intercept_unary_unary(self, continuation, client_call_details, request)
         64     grpc_request = {
         65         "payload": request_payload,
         66         "requestMethod": "grpc",
         67         "metadata": dict(request_metadata),
         68     }
         69     _LOGGER.debug(
         70         f"Sending request for {client_call_details.method}",
         71         extra={
       (...)     76         },
         77     )
    ---> 78 response = continuation(client_call_details, request)
         79 if logging_enabled:  # pragma: NO COVER
         80     response_metadata = response.trailing_metadata()


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/grpc/_interceptor.py:315, in _UnaryUnaryMultiCallable._with_call.<locals>.continuation(new_details, request)
        306 (
        307     new_method,
        308     new_timeout,
       (...)    312     new_compression,
        313 ) = _unwrap_client_call_details(new_details, client_call_details)
        314 try:
    --> 315     response, call = self._thunk(new_method).with_call(
        316         request,
        317         timeout=new_timeout,
        318         metadata=new_metadata,
        319         credentials=new_credentials,
        320         wait_for_ready=new_wait_for_ready,
        321         compression=new_compression,
        322     )
        323     return _UnaryOutcome(response, call)
        324 except grpc.RpcError as rpc_error:


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/grpc/_channel.py:1195, in _UnaryUnaryMultiCallable.with_call(self, request, timeout, metadata, credentials, wait_for_ready, compression)
       1183 def with_call(
       1184     self,
       1185     request: Any,
       (...)   1190     compression: Optional[grpc.Compression] = None,
       1191 ) -> Tuple[Any, grpc.Call]:
       1192     (
       1193         state,
       1194         call,
    -> 1195     ) = self._blocking(
       1196         request, timeout, metadata, credentials, wait_for_ready, compression
       1197     )
       1198     return _end_unary_response_blocking(state, call, True, None)


    File ~/Edu/modulab/llm-skt/.venv/lib/python3.12/site-packages/grpc/_channel.py:1162, in _UnaryUnaryMultiCallable._blocking(self, request, timeout, metadata, credentials, wait_for_ready, compression)
       1145 state.target = _common.decode(self._target)
       1146 call = self._channel.segregated_call(
       1147     cygrpc.PropagationConstants.GRPC_PROPAGATE_DEFAULTS,
       1148     self._method,
       (...)   1160     self._registered_call_handle,
       1161 )
    -> 1162 event = call.next_event()
       1163 _handle_event(event, state, self._response_deserializer)
       1164 return state, call


    File src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi:388, in grpc._cython.cygrpc.SegregatedCall.next_event()


    File src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi:211, in grpc._cython.cygrpc._next_call_event()


    File src/python/grpcio/grpc/_cython/_cygrpc/channel.pyx.pxi:205, in grpc._cython.cygrpc._next_call_event()


    File src/python/grpcio/grpc/_cython/_cygrpc/completion_queue.pyx.pxi:97, in grpc._cython.cygrpc._latent_event()


    File src/python/grpcio/grpc/_cython/_cygrpc/completion_queue.pyx.pxi:80, in grpc._cython.cygrpc._internal_latent_event()


    File src/python/grpcio/grpc/_cython/_cygrpc/completion_queue.pyx.pxi:61, in grpc._cython.cygrpc._next()


    KeyboardInterrupt: 


`(2) 커스텀 핸드오프 도구 사용`

- **핸드오프 도구**를 사용하여 에이전트 간 통신을 명시적으로 제어 (현재 에이전트에서 다음 에이전트로 이동하는 데 사용되는 도구)
- 이때, graph 인자를 **Command.PARENT**로 설정하여 부모 그래프로 돌아가기


```python
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

# 핸드오프 도구 생성 함수
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """커스텀 핸드오프 도구 생성"""
    name = f"transfer_to_{agent_name}"   # 다음에 이동할 에이전트 이름 (작업 할당)
    description = description or f"Transfer to {agent_name}"   # 에이전트 이동에 대한 설명 (작업 할당)

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState], 
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        
        # 도구 호출 메시지 생성
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )

        # 다음 에이전트로 이동하고 메시지 업데이트
        return Command(  
            goto=agent_name,  # 다음 에이전트로 이동 (작업 할당)
            update={"messages": state["messages"] + [tool_message]},   # 메시지 업데이트 (작업 할당)
            graph=Command.PARENT,  # 부모 그래프로 돌아가기 (작업 완료 후 슈퍼바이저에게 보고)
        )
    return handoff_tool
```


```python
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.chat_models import init_chat_model
# from langgraph_supervisor import create_handoff_tool


# 핸드오프 도구 (supervisor → research_agent)
transfer_to_research_agent = create_handoff_tool(
    agent_name="research_agent",
    description="정보 검색, 조사, 찾기 작업을 연구 전문가에게 할당"
)

# 핸드오프 도구 (supervisor → math_agent)
transfer_to_math_agent = create_handoff_tool(
    agent_name="math_agent", 
    description="수학 계산, 곱셈, 덧셈 작업을 수학 전문가에게 할당"
)

# 핸드오프 도구 (research_agent, math_agent → supervisor)
transfer_to_supervisor = create_handoff_tool(
    agent_name="supervisor",
    description="작업 완료 후 슈퍼바이저에게 보고"
)

# 슈퍼바이저 에이전트
supervisor = create_react_agent(
    model=init_chat_model("openai:gpt-4.1"),
    tools=[transfer_to_research_agent, transfer_to_math_agent],
    prompt="""당신은 팀 슈퍼바이저입니다. 사용자의 요청을 분석하여 적절한 전문가에게 작업을 할당하세요.

🔍 **연구 작업** (research_agent):
- 정보 검색, 조사, 찾기
- 웹 검색이 필요한 작업
- 데이터 수집

🧮 **수학 작업** (math_agent):  
- 계산, 곱셈, 덧셈, 나눗셈
- 수치 처리

사용자 요청을 분석하고 transfer_to_research_agent 또는 transfer_to_math_agent 도구를 사용하여 작업을 할당하세요.
복합 작업의 경우 먼저 정보 수집부터 시작하세요.""",
    name="supervisor"
)

# 연구 에이전트 (슈퍼바이저에게 복귀)
research_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[tavily_search, transfer_to_supervisor],
    prompt="""당신은 연구 전문가입니다. 웹 검색을 통해 정확한 정보를 찾아 제공하세요.

작업을 완료한 후에는 반드시 transfer_to_supervisor 도구를 사용해서 슈퍼바이저에게 결과를 보고하세요.""",
    name="research_agent"
)

# 수학 에이전트 (슈퍼바이저에게 복귀)
math_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[add, multiply, divide, transfer_to_supervisor],
    prompt="""당신은 수학 전문가입니다. 정확한 계산을 수행하세요.

작업을 완료한 후에는 반드시 transfer_to_supervisor 도구를 사용해서 슈퍼바이저에게 결과를 보고하세요.""",
    name="math_agent"
)

# 그래프 구성
supervisor_graph = (
    StateGraph(MessagesState)
    .add_node("supervisor", supervisor)
    .add_node("research_agent", research_agent)
    .add_node("math_agent", math_agent)
    .add_edge(START, "supervisor")  # 항상 슈퍼바이저부터 시작    
    .compile()
)

```


```python
# 실행
result = supervisor_graph.invoke({
    "messages": [{
        "role": "user", 
        "content": "한국의 인구를 찾아서, 인구 수에 2를 곱해주세요."
    }]
})

for m in result["messages"]:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    한국의 인구를 찾아서, 인구 수에 2를 곱해주세요.
    ==================================[1m Ai Message [0m==================================
    Name: supervisor
    Tool Calls:
      transfer_to_research_agent (call_cqoWtyp2duH5vzpMXMyc1ciu)
     Call ID: call_cqoWtyp2duH5vzpMXMyc1ciu
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_research_agent
    
    Successfully transferred to research_agent
    ==================================[1m Ai Message [0m==================================
    Name: research_agent
    Tool Calls:
      tavily_search (call_b6iW2ZYfZJoRDICwdu4x3d5x)
     Call ID: call_b6iW2ZYfZJoRDICwdu4x3d5x
      Args:
        query: 한국 인구
    =================================[1m Tool Message [0m=================================
    Name: tavily_search
    
    {"query": "한국 인구", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://namu.wiki/w/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD/%EC%9D%B8%EA%B5%AC", "title": "대한민국/인구 - 나무위키", "content": "대한민국의 인구를 정리한 문서다. 2025년 6월 기준으로 대한민국의 총 인구수는 51,164,582명이다. 이 중 남자 인구수는 25,467,115명이고,", "score": 0.8745175, "raw_content": null}, {"url": "https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EC%9D%B8%EA%B5%AC", "title": "대한민국의 인구 - 위키백과, 우리 모두의 백과사전", "content": "중위 추계에 따르면 한국은 2050년 65세 이상 노인 비율이 38.1%에 달하며, 같은 기간 일본 37.7%을 제치고 세계 1위 노인 비율 국가가 되며, 2060년에 이르러서는 65세 이상 노인 인구가 25~64세 일하는 인구보다 많아져 노인 부양이 사실상 어려울 것으로 예상된다. 저위 추계에 따르면, 한국은 65세 이상 노인 비율이 2045년 일본을 추월해 전 세계에서 가장 높은 국가가 될 것이며, 2100년 인구가 1928만 명으로 감소할 전망이다. | 독립국 |  네팔  대한민국  동티모르  라오스  러시아  레바논  말레이시아  몰디브  몽골  미얀마  바레인  방글라데시  베트남  부탄  브루나이  사우디아라비아  스리랑카  시리아  싱가포르  아랍에미리트  아르메니아  아제르바이잔  아프가니스탄  예멘  오만  요르단  우즈베키스탄  이라크  이란  이스라엘  이집트  인도  인도네시아  일본  조선민주주의인민공화국  조지아  중화민국  중화인민공화국  카자흐스탄  카타르  캄보디아  쿠웨이트  키르기스스탄  키프로스  타지키스탄  태국  투르크메니스탄  튀르키예  파키스탄  팔레스타인  필리핀 |  |", "score": 0.7769112, "raw_content": null}, {"url": "https://kosis.kr/visual/populationKorea/PopulationDashBoardMain.do", "title": "인구상황판 | 인구로 보는 대한민국 - KOSIS 국가통계포털", "content": "지역별 인구 통계에 대한 현황 및 정보를 확인할 수 있습니다. 관계도맵을 통해 인구와 사회·경제간 관계를 파악할 수 있습니다. 인구 통계를 더 다양한 형태로 경험해볼 수 있습니다. #### 40세 1972년생 또한 인구 구조가 출생, 사망, 이동 등 요인의 시나리오에 따라 어떻게 변화하는지 상세화면에서 더 살펴볼 수 있습니다. #### 40세 1972년생 또한 인구 구조가 출생, 사망, 이동 등 요인의 시나리오에 따라 어떻게 변화하는지 상세화면에서 더 살펴볼 수 있습니다. #### 40세 1972년생 또한 인구 구조가 출생, 사망, 이동 등 요인의 시나리오에 따라 어떻게 변화하는지 상세화면에서 더 살펴볼 수 있습니다. ### 총인구 (추계 기준) ### 학령인구 (6-21세) ### 1인 가구 ### 1인 가구 비중 ### 연령대별 1인 가구 (2025) ### 장래 출생아수 ### 장래 사망자수 ### 국제순이동 ### 외국인 국제순이동 ### 외국인 취업자", "score": 0.3507332, "raw_content": null}], "response_time": 1.18}
    ==================================[1m Ai Message [0m==================================
    Name: research_agent
    
    한국의 인구는 2025년 6월 기준으로 약 51,164,582명입니다. 이 인구 수에 2를 곱하면 약 102,329,164명이 됩니다.
    Tool Calls:
      transfer_to_supervisor (call_a7MBVcrfzbZI9rYtWmXlesoQ)
     Call ID: call_a7MBVcrfzbZI9rYtWmXlesoQ
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_supervisor
    
    Successfully transferred to supervisor
    ==================================[1m Ai Message [0m==================================
    Name: supervisor
    Tool Calls:
      transfer_to_math_agent (call_eBWCRe4Htt9SM7gcJ8543srA)
     Call ID: call_eBWCRe4Htt9SM7gcJ8543srA
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_math_agent
    
    Successfully transferred to math_agent
    ==================================[1m Ai Message [0m==================================
    Name: math_agent
    Tool Calls:
      multiply (call_gazZufAqfI99d0RfBORF209M)
     Call ID: call_gazZufAqfI99d0RfBORF209M
      Args:
        a: 51164582
        b: 2
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    102329164.0
    ==================================[1m Ai Message [0m==================================
    Name: math_agent
    Tool Calls:
      transfer_to_supervisor (call_rOu5Cdwi3ho8MeAI0rPPVeuB)
     Call ID: call_rOu5Cdwi3ho8MeAI0rPPVeuB
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_supervisor
    
    Successfully transferred to supervisor
    ==================================[1m Ai Message [0m==================================
    Name: supervisor
    
    한국의 인구(2025년 6월 기준)는 약 51,164,582명입니다.  
    이 인구 수에 2를 곱하면 102,329,164명이 됩니다.


### **2. Swarm 패턴** 

- **특징**: 분산형, 에이전트 간 자율적 협력
- **설치**
    - langgraph-swarm 패키지 사용

    ```bash
    pip install langgraph-swarm 
    ```

    ```bash
    uv add langgraph-swarm
    ```

![Swarm 패턴](https://langchain-ai.github.io/langgraph/agents/assets/swarm.png)


```python
from langgraph_swarm import create_swarm, create_handoff_tool

# 핸드오프 도구 생성
transfer_to_math_agent = create_handoff_tool(
    agent_name="math_agent",
    description="수학 계산이 필요할 때 수학 전문가에게 전달합니다."
)

transfer_to_research_agent = create_handoff_tool(
    agent_name="research_agent", 
    description="정보 검색이나 조사가 필요할 때 연구 전문가에게 전달합니다."
)

# 연구 에이전트 (핸드오프 도구 포함)
research_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[tavily_search, transfer_to_math_agent],
    prompt="""당신은 연구 전문가입니다. 웹 검색을 통해 정보를 찾아 제공합니다.
    
만약 검색 결과에서 숫자 데이터를 찾았고 사용자가 계산을 요청한다면, 
transfer_to_math_agent 도구를 사용해서 수학 전문가에게 작업을 전달하세요.""",
    name="research_agent"
)

# 수학 에이전트 (핸드오프 도구 포함)
math_agent = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[add, multiply, divide, transfer_to_research_agent],
    prompt="""당신은 수학 전문가입니다. 정확한 계산을 수행합니다.
    
만약 계산에 필요한 데이터가 없거나 추가 정보가 필요하다면,
transfer_to_research_agent 도구를 사용해서 연구 전문가에게 작업을 전달하세요.""",
    name="math_agent"
)

# 스웜 생성
swarm = create_swarm(
    agents=[research_agent, math_agent],
    default_active_agent="research_agent"  # 기본적으로 연구 에이전트가 먼저 시작
).compile()

# 시각화
display(Image(swarm.get_graph(xray=False).draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_004_LangGraph_Multi-Agent_18_0.png)
    



```python
# 실행
result = swarm.invoke({
    "messages": [{
        "role": "user", 
        "content": "한국의 인구를 찾아서, 인구 수에 2를 곱해주세요."
    }]
})

for m in result["messages"]:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    한국의 인구를 찾아서, 인구 수에 2를 곱해주세요.
    ==================================[1m Ai Message [0m==================================
    Name: research_agent
    Tool Calls:
      tavily_search (call_PQlzqh1IurwdqrwBOrzl4Gkb)
     Call ID: call_PQlzqh1IurwdqrwBOrzl4Gkb
      Args:
        query: 한국 인구 수
    =================================[1m Tool Message [0m=================================
    Name: tavily_search
    
    {"query": "한국 인구 수", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://namu.wiki/w/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD/%EC%9D%B8%EA%B5%AC", "title": "대한민국/인구 - 나무위키", "content": "대한민국의 인구를 정리한 문서다. 2025년 6월 기준으로 대한민국의 총 인구수는 51,164,582명이다. 이 중 남자 인구수는 25,467,115명이고,", "score": 0.89166987, "raw_content": null}, {"url": "https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EC%9D%B8%EA%B5%AC", "title": "대한민국의 인구 - 위키백과, 우리 모두의 백과사전", "content": "중위 추계에 따르면 한국은 2050년 65세 이상 노인 비율이 38.1%에 달하며, 같은 기간 일본 37.7%을 제치고 세계 1위 노인 비율 국가가 되며, 2060년에 이르러서는 65세 이상 노인 인구가 25~64세 일하는 인구보다 많아져 노인 부양이 사실상 어려울 것으로 예상된다. 저위 추계에 따르면, 한국은 65세 이상 노인 비율이 2045년 일본을 추월해 전 세계에서 가장 높은 국가가 될 것이며, 2100년 인구가 1928만 명으로 감소할 전망이다. | 독립국 |  네팔  대한민국  동티모르  라오스  러시아  레바논  말레이시아  몰디브  몽골  미얀마  바레인  방글라데시  베트남  부탄  브루나이  사우디아라비아  스리랑카  시리아  싱가포르  아랍에미리트  아르메니아  아제르바이잔  아프가니스탄  예멘  오만  요르단  우즈베키스탄  이라크  이란  이스라엘  이집트  인도  인도네시아  일본  조선민주주의인민공화국  조지아  중화민국  중화인민공화국  카자흐스탄  카타르  캄보디아  쿠웨이트  키르기스스탄  키프로스  타지키스탄  태국  투르크메니스탄  튀르키예  파키스탄  팔레스타인  필리핀 |  |", "score": 0.6894943, "raw_content": null}, {"url": "https://kosis.kr/visual/populationKorea/PopulationDashBoardMain.do", "title": "인구상황판 | 인구로 보는 대한민국 - KOSIS 국가통계포털", "content": "지역별 인구 통계에 대한 현황 및 정보를 확인할 수 있습니다. 관계도맵을 통해 인구와 사회·경제간 관계를 파악할 수 있습니다. 인구 통계를 더 다양한 형태로 경험해볼 수 있습니다. #### 40세 1972년생 또한 인구 구조가 출생, 사망, 이동 등 요인의 시나리오에 따라 어떻게 변화하는지 상세화면에서 더 살펴볼 수 있습니다. #### 40세 1972년생 또한 인구 구조가 출생, 사망, 이동 등 요인의 시나리오에 따라 어떻게 변화하는지 상세화면에서 더 살펴볼 수 있습니다. #### 40세 1972년생 또한 인구 구조가 출생, 사망, 이동 등 요인의 시나리오에 따라 어떻게 변화하는지 상세화면에서 더 살펴볼 수 있습니다. ### 총인구 (추계 기준) ### 학령인구 (6-21세) ### 1인 가구 ### 1인 가구 비중 ### 연령대별 1인 가구 (2025) ### 장래 출생아수 ### 장래 사망자수 ### 국제순이동 ### 외국인 국제순이동 ### 외국인 취업자", "score": 0.2750377, "raw_content": null}], "response_time": 0.96}
    ==================================[1m Ai Message [0m==================================
    Name: research_agent
    Tool Calls:
      transfer_to_math_agent (call_j4PhPXMq9VhgG8PqYbvUQz4s)
     Call ID: call_j4PhPXMq9VhgG8PqYbvUQz4s
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_math_agent
    
    Successfully transferred to math_agent
    ==================================[1m Ai Message [0m==================================
    Name: math_agent
    Tool Calls:
      multiply (call_zEvaMRfYvYpnVLz768SAzpcO)
     Call ID: call_zEvaMRfYvYpnVLz768SAzpcO
      Args:
        a: 51164582
        b: 2
    =================================[1m Tool Message [0m=================================
    Name: multiply
    
    102329164.0
    ==================================[1m Ai Message [0m==================================
    Name: math_agent
    
    한국의 인구는 약 51,164,582명이고, 이 인구 수에 2를 곱하면 102,329,164명이 됩니다.


### **3. 계층적 아키텍처 (Supervisor + Swarm 조합)** 

- **특징**: 슈퍼바이저와 스웜을 조합하여 복잡한 작업 흐름 구현
- **적용**: 대규모 시스템에서 에이전트 팀들을 관리할 때 (상위 슈퍼바이저가 하위 스웜들을 관리하는 계층적 시스템)


```python
from langgraph_supervisor import create_supervisor
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

# 하위 스웜 1: 정보 수집 팀
research_handoff = create_handoff_tool(
    agent_name="data_analyst",
    description="상세한 데이터 분석이 필요할 때 데이터 분석가에게 전달"
)

basic_researcher = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[tavily_search, research_handoff],
    prompt="기본 연구자. 웹 검색으로 정보 수집. 상세 분석이 필요하면 데이터 분석가에게 전달.",
    name="basic_researcher"
)

analyst_handoff = create_handoff_tool(
    agent_name="basic_researcher", 
    description="기본 정보 검색이 필요할 때 기본 연구자에게 전달"
)

data_analyst = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[tavily_search, analyst_handoff],
    prompt="데이터 분석가. 상세한 분석과 인사이트 제공. 기본 검색이 필요하면 기본 연구자에게 전달.",
    name="data_analyst"
)

research_swarm = create_swarm(
    agents=[basic_researcher, data_analyst],
    default_active_agent="basic_researcher"
).compile(name="research_swarm")

# 하위 스웜 2: 계산 팀  
calc_handoff = create_handoff_tool(
    agent_name="advanced_calculator",
    description="복잡한 계산이 필요할 때 고급 계산기에게 전달"
)

basic_calculator = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[add, multiply, calc_handoff],
    prompt="기본 계산기. 간단한 덧셈과 곱셈 수행. 복잡한 계산은 고급 계산기에게 전달.",
    name="basic_calculator"
)

advanced_handoff = create_handoff_tool(
    agent_name="basic_calculator",
    description="기본 계산이 필요할 때 기본 계산기에게 전달"
)

advanced_calculator = create_react_agent(
    model=init_chat_model("openai:gpt-4.1-mini"),
    tools=[add, multiply, divide, advanced_handoff],
    prompt="고급 계산기. 복잡한 계산 수행. 기본 계산은 기본 계산기에게 전달.",
    name="advanced_calculator"
)

calc_swarm = create_swarm(
    agents=[basic_calculator, advanced_calculator],
    default_active_agent="basic_calculator"
).compile(name="calc_swarm")

# 최상위 슈퍼바이저
top_supervisor = create_supervisor(
    agents=[research_swarm, calc_swarm],
    model=init_chat_model("openai:gpt-4.1"),
    prompt="""최상위 슈퍼바이저입니다. 두 개의 전문 팀을 관리합니다:

1. research_swarm: 정보 검색과 데이터 분석 담당
2. calc_swarm: 수학 계산 담당

작업의 성격에 따라 적절한 팀에게 할당하세요."""
).compile(name="top_supervisor")


# 그래프 시각화
display(Image(top_supervisor.get_graph(xray=False).draw_mermaid_png()))
```


    
![png](/Users/jussuit/Desktop/temp/data/processed/markdown/day6/DAY06_004_LangGraph_Multi-Agent_21_0.png)
    



```python
# 실행
result = top_supervisor.invoke({
    "messages": [{
        "role": "user", 
        "content": "한국의 인구를 찾아서, 인구 수에 2를 곱해주세요."
    }]
})

for m in result["messages"]:
    m.pretty_print()
```

    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel is_last_step, ignoring it.
    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel remaining_steps, ignoring it.
    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel is_last_step, ignoring it.
    Task supervisor with path ('__pregel_pull', 'supervisor') wrote to unknown channel remaining_steps, ignoring it.


    ================================[1m Human Message [0m=================================
    
    한국의 인구를 찾아서, 인구 수에 2를 곱해주세요.
    ==================================[1m Ai Message [0m==================================
    Name: supervisor
    Tool Calls:
      transfer_to_research_swarm (call_xireom4WNo7wL3fL0QlN502P)
     Call ID: call_xireom4WNo7wL3fL0QlN502P
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_research_swarm
    
    Successfully transferred to research_swarm
    ==================================[1m Ai Message [0m==================================
    Name: basic_researcher
    
    2025년 6월 기준 한국의 총 인구수는 약 51,164,582명입니다. 이를 2배 하면 102,329,164명입니다.
    ==================================[1m Ai Message [0m==================================
    Name: research_swarm
    
    Transferring back to supervisor
    Tool Calls:
      transfer_back_to_supervisor (f753c531-768f-4c4d-9750-cc499b5ec465)
     Call ID: f753c531-768f-4c4d-9750-cc499b5ec465
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_back_to_supervisor
    
    Successfully transferred back to supervisor
    ==================================[1m Ai Message [0m==================================
    Name: supervisor
    Tool Calls:
      transfer_to_calc_swarm (call_MX60Vf7yaI7goKuUYUMjqLub)
     Call ID: call_MX60Vf7yaI7goKuUYUMjqLub
      Args:
        korean_population: 51164582
        multiplier: 2
    =================================[1m Tool Message [0m=================================
    Name: transfer_to_calc_swarm
    
    Successfully transferred to calc_swarm
    ==================================[1m Ai Message [0m==================================
    Name: basic_calculator
    
    한국의 인구 약 51,164,582명에 2를 곱하면 102,329,164명입니다.
    ==================================[1m Ai Message [0m==================================
    Name: calc_swarm
    
    Transferring back to supervisor
    Tool Calls:
      transfer_back_to_supervisor (2692c18b-5dea-4e4e-96bf-69715c1fa147)
     Call ID: 2692c18b-5dea-4e4e-96bf-69715c1fa147
      Args:
    =================================[1m Tool Message [0m=================================
    Name: transfer_back_to_supervisor
    
    Successfully transferred back to supervisor
    ==================================[1m Ai Message [0m==================================
    Name: supervisor
    
    2025년 6월 기준 한국의 인구는 약 51,164,582명입니다.  
    이 수에 2를 곱하면 102,329,164명입니다.



```python

```
