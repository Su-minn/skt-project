#!/usr/bin/env python3
"""
마크다운 파일 단계별 LLM 정제 스크립트
 
기능:
- 1단계: 기본 정제 (환경 설정, 이미지, 실행 결과, 디버깅 코드 제거)
- 2단계: 계층 구조 조정
- 3단계: 개념 설명 개선  
- 4단계: 개념 간 관계 명시화
 
사용법:
    python scripts/markdown_llm_refiner.py <input_file> [output_file]
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class MarkdownLLMRefiner:
    def __init__(self, openai_model_name: str = "gpt-4.1", google_model_name: str = "gemini-2.0-flash"):
        self.openai_llm = ChatOpenAI(model=openai_model_name, temperature=0.1)
        self.google_llm = ChatGoogleGenerativeAI(model=google_model_name, temperature=0.1)
    
    def step1_basic_cleanup(self, content: str) -> str:
        """1단계: 기본 정제 - 환경 설정, 이미지, 디버깅 코드 제거"""
        print("🧹 1단계: 기본 정제 시작...")
        
        system_prompt = """
당신은 LangGraph 학습 자료의 마크다운 기본 정제 전문가입니다.

다음 작업만 수행해주세요:

## 🧹 기본 정제 (자동 제거)

**환경 설정 제거**:
- "환경 설정 및 준비" 섹션 전체 제거
- "Env 환경변수", "기본 라이브러리", "langfuse", "콜백 핸들러" 관련 섹션 제거
- import 문 중 환경 설정 관련만 제거 (핵심 기능 import는 유지)

**불필요한 요소 제거**:
- 모든 이미지 태그 제거 (![...](...), <img>)
- 실행 결과 중 의미 없는 출력 제거 (단순 True/False, 딕셔너리 출력)
- 디버깅용 코드 제거 (langfuse_handler, display(Image), 테스트용 주석)
- 빈 코드 블록 및 연속된 빈 줄 정리

## ⚠️ 주의사항
- 핵심 정보와 코드는 절대 삭제하지 말고 그대로 유지
- 학습에 필요한 import 문은 유지
- 실습 코드와 의미 있는 결과는 보존
- 원본의 내용과 구조는 최대한 보존하고 불필요한 부분만 제거
"""
        
        human_prompt = f"""
다음 LangGraph 학습 자료에서 환경 설정, 이미지, 디버깅 코드 등 불필요한 요소만 제거해주세요:

{content}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.google_llm.invoke(messages)
        print("✅ 1단계: 기본 정제 완료")
        return response.content
    
    def step2_structure_improvement(self, content: str) -> str:
        """2단계: 구조 개선 - 계층 조정 및 내용 구조화"""
        print("📋 2단계: 구조 개선 시작...")
        
        system_prompt = """
당신은 LangGraph 학습 자료의 마크다운 구조를 개선하는 전문가입니다.

다음 작업만 수행해주세요:

## 📋 구조 개선

**계층 구조 조정**:
- 핵심 개념 (StateGraph, Command 등)을 ## 레벨로
- 하위 개념 (State, Node, Graph 등)을 ### 레벨로  
- 구현 세부사항을 #### 레벨로
- 논리적 학습 순서: 기초 개념 → 심화 개념 → 실습 → 도구

**내용 구조화**:
- 개념 설명과 코드를 명확히 분리
- 각 섹션의 목적을 명시
- 일관된 표기법 사용

## ⚠️ 주의사항
- 내용은 변경하지 말고 구조와 계층만 조정
- 모든 원본 정보 보존
- 학습 흐름 유지
"""
        
        human_prompt = f"""
다음 마크다운의 계층 구조와 내용 구조를 개선해주세요:

{content}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.openai_llm.invoke(messages)
        print("✅ 2단계: 구조 개선 완료")
        return response.content
    
    def step3_concept_improvement(self, content: str) -> str:
        """3단계: 개념 설명 개선"""
        print("📝 3단계: 개념 설명 개선 시작...")
        
        system_prompt = """
당신은 LangGraph 기술 문서 작성 전문가입니다.

다음 작업만 수행해주세요:

## 📝 개념 설명 개선

각 주요 개념을 다음 형식으로 정리:

```markdown
## [개념명]

**정의**: [명확한 정의]

**특징**:
- [주요 특징 1]
- [주요 특징 2]

**활용**: [어떤 상황에서 사용되는지]

### 코드 예제
[기존 코드와 설명 유지]

### 실행 결과
[실행 결과 유지]
```

## ⚠️ 주의사항
- 기존 코드와 예제는 그대로 유지
- 설명만 명확하고 일관되게 개선
- 새로운 내용 추가 금지
"""
        
        human_prompt = f"""
다음 마크다운의 개념 설명을 위 형식에 맞게 개선해주세요:

{content}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.google_llm.invoke(messages)
        print("✅ 3단계: 개념 설명 개선 완료")
        return response.content
    
    def step4_relationship_mapping(self, content: str) -> str:
        """4단계: 개념 간 관계 명시화"""
        print("🔗 4단계: 관계 명시화 시작...")
        
        system_prompt = """
당신은 LangGraph 지식 그래프 설계 전문가입니다.

다음 작업만 수행해주세요:

## 🔗 관계 명시화

각 개념에 다음 관계 정보를 추가:

**선행 개념**: [이 개념을 이해하기 위해 먼저 알아야 할 개념들]
**연관 개념**: [함께 사용되거나 관련된 개념들]

관계 유형:
- PREREQUISITE: 선행 학습 필요
- BUILDS_UPON: 기반으로 확장
- IMPLEMENTS: 코드가 개념 구현
- TEACHES: 튜토리얼이 개념 가르침
- SUPPORTS: 도구가 개념 지원

## ⚠️ 주의사항
- 기존 내용은 그대로 유지
- 각 개념 섹션에 관계 정보만 추가
- 실제 문서에 나타난 개념들 간의 관계만 명시
"""
        
        human_prompt = f"""
다음 마크다운에서 개념 간 관계를 분석하고 명시해주세요:

{content}
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        response = self.google_llm.invoke(messages)
        print("✅ 4단계: 관계 명시화 완료")
        return response.content
    
    def refine_step_by_step(self, content: str) -> str:
        """4단계에 걸쳐 순차적으로 정제를 수행합니다."""
        original_lines = len(content.splitlines())
        original_size = len(content)
        
        print(f"📊 원본: {original_lines}줄, {original_size:,}자")
        print("=" * 50)
        
        # 1단계: 기본 정제
        content = self.step1_basic_cleanup(content)
        step1_lines = len(content.splitlines())
        step1_size = len(content)
        print(f"📊 1단계 후: {step1_lines}줄 ({step1_lines-original_lines:+d}), {step1_size:,}자 ({((step1_size-original_size)/original_size)*100:+.1f}%)")
        print("=" * 50)
        
        # 2단계: 구조 개선  
        content = self.step2_structure_improvement(content)
        step2_lines = len(content.splitlines())
        step2_size = len(content)
        print(f"📊 2단계 후: {step2_lines}줄 ({step2_lines-step1_lines:+d}), {step2_size:,}자 ({((step2_size-step1_size)/step1_size)*100:+.1f}%)")
        print("=" * 50)
        
        # 3단계: 개념 설명 개선
        content = self.step3_concept_improvement(content)
        step3_lines = len(content.splitlines())
        step3_size = len(content)
        print(f"📊 3단계 후: {step3_lines}줄 ({step3_lines-step2_lines:+d}), {step3_size:,}자 ({((step3_size-step2_size)/step2_size)*100:+.1f}%)")
        print("=" * 50)
        
        # 4단계: 관계 명시화
        content = self.step4_relationship_mapping(content)
        final_lines = len(content.splitlines())
        final_size = len(content)
        print(f"📊 4단계 후: {final_lines}줄 ({final_lines-step3_lines:+d}), {final_size:,}자 ({((final_size-step3_size)/step3_size)*100:+.1f}%)")
        print("=" * 50)
        
        # 최종 요약
        total_lines_change = final_lines - original_lines
        total_size_change = final_size - original_size
        total_size_change_percent = (total_size_change / original_size) * 100 if original_size > 0 else 0
        
        print(f"\n📊 최종 요약:")
        print(f"   원본 → 최종: {original_lines}줄 → {final_lines}줄 ({total_lines_change:+d}줄)")
        print(f"   크기 변화: {original_size:,}자 → {final_size:,}자 ({total_size_change_percent:+.1f}%)")
        
        return content

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python scripts/markdown_llm_refiner.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_suffix('.refined.md')
    
    if not input_file.exists():
        print(f"❌ 입력 파일이 없습니다: {input_file}")
        sys.exit(1)
    
    print(f"📄 입력 파일: {input_file}")
    print(f"📄 출력 파일: {output_file}")
    print()
    
    try:
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 단계별 정제
        print("🤖 단계별 LLM 정제 시작...")
        refiner = MarkdownLLMRefiner()
        refined_content = refiner.refine_step_by_step(content)
        
        # 파일 저장
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(refined_content)
        
        print(f"\n🎉 단계별 정제 완료!")
        print(f"💾 저장 위치: {output_file}")
        print(f"\n✅ 완료된 단계:")
        print(f"  1️⃣ 기본 정제: 환경 설정, 이미지, 디버깅 코드 제거")
        print(f"  2️⃣ 구조 개선: 계층 조정, 내용 구조화")
        print(f"  3️⃣ 개념 개선: 정의, 특징, 활용 명확화")
        print(f"  4️⃣ 관계 명시: 선행/연관 개념 매핑")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
