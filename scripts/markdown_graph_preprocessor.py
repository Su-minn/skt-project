#!/usr/bin/env python3
"""
LangGraph 마크다운 파일 Graph DB 전처리 스크립트

기능:
- 마크다운 파일을 Graph DB 파싱용 구조화된 데이터로 변환
- 노드 타입별 분류 (Concept, Component, CodeExample, Tutorial)
- 노드 간 관계(Relationship) 추출
- JSON 형태로 구조화된 결과 저장

사용법:
    python scripts/markdown_graph_preprocessor.py <input_file> [output_file]
    python scripts/markdown_graph_preprocessor.py --batch <input_directory> <output_directory>
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


class LLMGraphPreprocessor:
    """LangGraph 학습 자료를 Graph DB 파싱용으로 전처리하는 클래스"""
    
    def __init__(self, model_name: str = "gpt-4.1"):
        """
        Args:
            model_name: 사용할 OpenAI 모델명 (기본값: gpt-4.1)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.preprocessing_prompt = self._get_preprocessing_prompt()
    
    def _get_preprocessing_prompt(self) -> str:
        """Graph DB 파싱용 전처리 프롬프트 반환"""
        return """
당신은 LangGraph 학습 자료를 Graph DB 파싱을 위해 전처리하는 전문가입니다.

## 중요 규칙
1. **중복 금지**: 같은 이름은 Concept 또는 Component 중 하나에만 속해야 합니다
2. **명확한 구분**: 
   - Concept = 추상적 개념, 이론 (예: "상태 관리", "그래프 구조")
   - Component = 실제 코드 클래스/함수 (예: StateGraph 클래스, ChatOpenAI 클래스)
3. **일관된 네이밍**: 관계에서는 이름만 사용 (괄호 없이)

## 노드 타입 정의

### Concept (추상 개념)
- 정의: 이론적 개념, 패턴, 원리
- 예시: "State Management", "Graph Structure", "Conditional Routing"
- 주의: 실제 클래스명은 Component로 분류

### Component (구현체)
- 정의: import 가능한 클래스, 함수
- 예시: StateGraph, ChatOpenAI, TypedDict, Command
- 주의: 코드에서 직접 사용하는 클래스/함수명

### CodeExample (코드 예시)
- 정의: 특정 기능을 보여주는 코드 조각
- 주의: title은 고유해야 함

### Tutorial (실습)
- 정의: 완전한 시나리오를 구현하는 실습
- 주의: [실습] 섹션이나 전체 구현 예제

## 출력 형식

### CONCEPTS
```
[Concept]
name: State Management
description: 그래프에서 데이터를 관리하고 노드 간 공유하는 개념
source: {파일명}
```

### COMPONENTS
```
[Component]
name: StateGraph
description: LangGraph에서 상태 기반 워크플로우를 구현하는 클래스
source: {파일명}
```

### CODE_EXAMPLES (단순한 사용 예제)
```
[CodeExample]
title: 기본 StateGraph 생성
description: StateGraph 인스턴스를 생성하고 설정하는 기본 예제
code: [여기에 단일 기능의 코드 블록 - 5-20줄, 특정 컴포넌트의 기본 사용법]
source: {파일명}
```

#### CodeExample 판별 기준:
- **단일 컴포넌트**의 기본 사용법
- **간단한 설정**이나 초기화 
- **특정 기능**의 데모나 테스트
- **짧고 단순한** 코드 (5-20줄)
- 예: "기본 설정", "단순 호출", "초기화 예제"

### TUTORIALS (복합적인 전체 구현)
```
[Tutorial]  
title: 메모리 기반 대화 시스템 구축
description: StateGraph와 Memory를 조합하여 상태 유지 대화 시스템을 처음부터 끝까지 구현하는 전체 튜토리얼
code: [여기에 완전한 end-to-end 구현 - 20-100줄, 전체 워크플로우와 실행 예시]
source: {파일명}
```

#### Tutorial 판별 기준:
- **여러 컴포넌트**를 조합한 복합 시스템
- **전체 워크플로우**나 **완전한 구현**
- **실제 문제 해결**을 위한 end-to-end 예제
- **단계별 과정**을 보여주는 실습
- **긴 구현** (20줄 이상)
- 예: "시스템 구축", "전체 구현", "실습", "워크플로우"

### RELATIONSHIPS
```
[Relationship]
type: IMPLEMENTS
from: StateGraph
to: State Management

[Relationship]
type: USES
from: 기본 StateGraph 생성
to: StateGraph

[Relationship]
type: APPLIES
from: 메모리 기반 대화 시스템
to: Memory Management

[Relationship]
type: REQUIRES
from: Multi-Agent Systems
to: State Management

[Relationship]
type: INCLUDES
from: 메모리 기반 대화 시스템
to: InMemorySaver
```

## 관계 규칙 (매우 중요!)

### 필수 관계 타입 5가지

1. **IMPLEMENTS**: Component → Concept
   - Component가 어떤 개념을 구현하는지
   - 예: StateGraph -IMPLEMENTS-> State Management

2. **USES**: CodeExample → Component  
   - 코드 예제가 어떤 컴포넌트를 사용하는지
   - 예: 기본 StateGraph 생성 -USES-> StateGraph

3. **APPLIES**: Tutorial → Concept
   - 튜토리얼이 어떤 개념을 다루는지  
   - 예: 메모리 기반 대화 시스템 -APPLIES-> Memory Management

4. **REQUIRES**: Concept → Concept
   - 학습 순서상 선행 지식 관계
   - 예: Multi-Agent -REQUIRES-> State Management

5. **INCLUDES**: Tutorial → Component
   - 튜토리얼에서 사용된 주요 컴포넌트
   - 예: 메모리 기반 대화 시스템 -INCLUDES-> InMemorySaver

### 관계 추출 규칙
- **모든 5가지 관계 타입을 반드시 찾아서 추출**
- from/to에는 정확한 이름만 사용 (괄호, 설명 없음)
- 각 노드마다 최소 1-3개의 관계를 찾을 것
- 컨텍스트를 분석하여 의미론적 관계를 추론할 것

## 코드 추출 규칙 (매우 중요!)

### CodeExample 코드 추출
- **전체 코드 블록** 추출: 마크다운의 ```python 블록 전체를 포함
- **함수 정의 포함**: def, class 등 전체 정의
- **import 문 포함**: 필요한 모든 import
- **실행 코드 포함**: 실제 호출, 인스턴스 생성 등
- **최소 5-20줄**: 단순 1줄이 아닌 의미있는 코드 블록

### Tutorial 코드 추출  
- **완전한 구현**: 처음부터 끝까지 전체 코드
- **모든 함수**: 상태 정의, 노드 함수, 그래프 구성 등
- **실행 예시**: 마지막에 실행하는 코드까지 포함
- **최소 20-100줄**: 완전한 튜토리얼 코드

### 추가 규칙
- **source 정제**: 파일명에서 .refined, .processed 등 접미사 제거
- **여러 줄 처리**: 줄바꿈은 \\n으로 하나의 문자열에 저장
- **코드 우선**: description보다 실제 코드가 더 중요

**반드시 마크다운의 코드 블록(```python)을 찾아서 전체를 추출하세요!**

## 코드 출력 방식
CodeExample과 Tutorial의 code 필드에는:
1. 해당하는 마크다운 코드 블록의 **전체 내용**을 복사
2. 여러 줄 코드는 줄바꿈(\n)으로 연결하여 하나의 문자열로 작성
3. 예시: "import os\nfrom typing import Dict\n\ndef my_function():\n    return 'hello'"

## 분류 지침 (중요!)

### Tutorial vs CodeExample 분류할 때:

1. **제목과 내용 분석**:
   - Tutorial 키워드: "시스템", "구축", "전체", "실습", "워크플로우", "구현", "조합"
   - CodeExample 키워드: "기본", "간단한", "설정", "초기화", "사용법", "예제", "데모"

2. **복잡도 확인**:
   - 여러 컴포넌트를 조합? → Tutorial
   - 단일 기능만 보여줌? → CodeExample

3. **코드 길이**:
   - 20줄 이상의 복합 구현? → Tutorial  
   - 20줄 미만의 간단한 예제? → CodeExample

**반드시 위 기준에 따라 정확히 분류하세요!**

위 형식을 정확히 따라 전처리하세요.
"""
    
    def preprocess_markdown(self, markdown_content: str, filename: str) -> Dict[str, Any]:
        """
        마크다운 파일을 Graph DB 파싱용 구조로 전처리
        
        Args:
            markdown_content: 마크다운 파일 내용
            filename: 파일명 (확장자 제외)
            
        Returns:
            구조화된 전처리 결과 딕셔너리
        """
        print(f"🔄 {filename} 전처리 중...")
        
        # source 이름 정제
        clean_filename = self._clean_source_name(filename)
        
        messages = [
            SystemMessage(content=self.preprocessing_prompt),
            HumanMessage(content=f"""
파일명: {clean_filename}

다음 마크다운 내용을 전처리해주세요:

{markdown_content}
""")
        ]
        
        try:
            response = self.llm.invoke(messages)
            result = self._parse_llm_response(response.content, clean_filename)
            
            # 후처리 추가
            result = self._post_process_result(result)
            
            return result
            
        except Exception as e:
            print(f"❌ LLM 처리 중 오류: {e}")
            return self._create_empty_result(clean_filename)
    
    def _parse_llm_response(self, response: str, filename: str) -> Dict[str, Any]:
        """LLM 응답을 구조화된 데이터로 파싱"""
        
        result = {
            'source_file': filename,
            'concepts': [],
            'components': [],
            'code_examples': [],
            'tutorials': [],
            'relationships': []
        }
        
        # 이름 추적을 위한 집합
        concept_names = set()
        component_names = set()
        
        current_section = None
        current_item = {}
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 섹션 구분
            if line == '### CONCEPTS':
                current_section = 'concepts'
                continue
            elif line == '### COMPONENTS':
                current_section = 'components'
                continue
            elif line == '### CODE_EXAMPLES':
                current_section = 'code_examples'
                continue
            elif line == '### TUTORIALS':
                current_section = 'tutorials'
                continue
            elif line == '### RELATIONSHIPS':
                current_section = 'relationships'
                continue
            
            # 아이템 시작 감지
            if line.startswith('[Concept]'):
                if current_item:
                    self._add_item_to_result(result, current_section, current_item, 
                                           concept_names, component_names)
                current_item = {'type': 'Concept'}
            elif line.startswith('[Component]'):
                if current_item:
                    self._add_item_to_result(result, current_section, current_item,
                                           concept_names, component_names)
                current_item = {'type': 'Component'}
            elif line.startswith('[CodeExample]'):
                if current_item:
                    self._add_item_to_result(result, current_section, current_item,
                                           concept_names, component_names)
                current_item = {'type': 'CodeExample'}
            elif line.startswith('[Tutorial]'):
                if current_item:
                    self._add_item_to_result(result, current_section, current_item,
                                           concept_names, component_names)
                current_item = {'type': 'Tutorial'}
            elif line.startswith('[Relationship]'):
                if current_item:
                    self._add_item_to_result(result, current_section, current_item,
                                           concept_names, component_names)
                current_item = {'type': 'Relationship'}
            
            # 속성 파싱
            elif ':' in line and current_item:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 관계의 from/to에서 괄호 제거
                if key in ['from', 'to'] and '(' in value:
                    value = value.split('(')[0].strip()
                
                # source 정제 (확장자 및 접미사 제거)
                if key == 'source':
                    value = self._clean_source_name(value)
                
                if key in ['name', 'title', 'description', 'source', 'type', 'from', 'to', 'code']:
                    current_item[key] = value
            
            # code 속성의 경우 여러 줄 처리
            elif current_item and 'code' in current_item and line and not line.startswith('['):
                # 기존 code에 새 줄 추가
                current_item['code'] += '\n' + line
        
        # 마지막 아이템 추가
        if current_item and current_section:
            self._add_item_to_result(result, current_section, current_item,
                                   concept_names, component_names)
        
        # 결과 검증 및 정제
        result = self._validate_and_clean_result(result)
        
        return result
    
    def _add_item_to_result(self, result: Dict, section: str, item: Dict, 
                           concept_names: set, component_names: set):
        """아이템을 결과에 추가 (중복 체크 포함)"""
        if not section or not item or section not in result:
            return
        
        # type 속성이 잘못된 경우 수정
        if section == 'components' and item.get('type') == 'Concept':
            item['type'] = 'Component'
        
        # 중복 체크
        if section == 'concepts':
            name = item.get('name')
            if name and name not in component_names:  # Component에 없을 때만 추가
                concept_names.add(name)
                if self._validate_item(item, section):
                    result[section].append(item.copy())
        elif section == 'components':
            name = item.get('name')
            if name and name not in concept_names:  # Concept에 없을 때만 추가
                component_names.add(name)
                if self._validate_item(item, section):
                    result[section].append(item.copy())
        else:
            # 다른 섹션은 일반 검증만
            if self._validate_item(item, section):
                result[section].append(item.copy())
    
    def _validate_item(self, item: Dict, section: str) -> bool:
        """아이템 필수 필드 검증"""
        if section == 'relationships':
            return all(key in item for key in ['type', 'from', 'to'])
        else:
            # concepts, components, code_examples, tutorials
            required_fields = ['description', 'source']
            if section in ['code_examples', 'tutorials']:
                required_fields.extend(['title', 'code'])  # code 필드 필수
            else:
                required_fields.append('name')
            
            return all(key in item and item[key].strip() for key in required_fields)
    
    def _validate_and_clean_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """결과 검증 및 정제"""
        
        # 중복 제거
        for section in ['concepts', 'components', 'code_examples', 'tutorials']:
            if section in result:
                seen = set()
                unique_items = []
                for item in result[section]:
                    identifier = item.get('name') or item.get('title', '')
                    if identifier and identifier not in seen:
                        seen.add(identifier)
                        unique_items.append(item)
                result[section] = unique_items
        
        # 통계 정보 추가
        stats = {}
        for section in ['concepts', 'components', 'code_examples', 'tutorials', 'relationships']:
            stats[section] = len(result.get(section, []))
        
        result['statistics'] = stats
        
        return result
    
    def _clean_source_name(self, source_name: str) -> str:
        """source 이름 정제 - 확장자 및 접미사 제거"""
        # 확장자 제거
        if '.' in source_name:
            source_name = source_name.rsplit('.', 1)[0]
        
        # 접미사 제거 (.refined, .processed, .cleaned 등)
        suffixes_to_remove = ['.refined', '.processed', '.cleaned', '.final', '.updated']
        for suffix in suffixes_to_remove:
            if source_name.endswith(suffix):
                source_name = source_name[:-len(suffix)]
                break
        
        return source_name
    
    def _post_process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """결과 후처리 - 중복 제거 및 일관성 확보"""
        
        # 1. Concept과 Component 중복 제거
        concept_names = {c['name'] for c in result.get('concepts', [])}
        component_names = {c['name'] for c in result.get('components', [])}
        
        # 중복된 이름 찾기
        duplicates = concept_names & component_names
        
        if duplicates:
            print(f"⚠️  중복 발견: {duplicates}")
            # Component로 통일 (실제 클래스명인 경우가 많음)
            result['concepts'] = [c for c in result['concepts'] 
                                if c['name'] not in duplicates]
        
        # 2. 관계 정리
        cleaned_relationships = []
        for rel in result.get('relationships', []):
            # from/to에서 괄호 내용 제거
            if 'from' in rel:
                rel['from'] = rel['from'].split('(')[0].strip()
            if 'to' in rel:
                rel['to'] = rel['to'].split('(')[0].strip()
            
            # 유효한 관계만 추가
            if self._validate_relationship(rel, result):
                cleaned_relationships.append(rel)
        
        result['relationships'] = cleaned_relationships
        
        return result
    
    def _validate_relationship(self, rel: Dict, result: Dict) -> bool:
        """관계 유효성 검증"""
        from_name = rel.get('from', '')
        to_name = rel.get('to', '')
        rel_type = rel.get('type', '')
        
        # 모든 노드의 이름/타이틀 수집
        all_names = set()
        
        for concept in result.get('concepts', []):
            all_names.add(concept.get('name', ''))
        
        for component in result.get('components', []):
            all_names.add(component.get('name', ''))
        
        for example in result.get('code_examples', []):
            all_names.add(example.get('title', ''))
        
        for tutorial in result.get('tutorials', []):
            all_names.add(tutorial.get('title', ''))
        
        # from과 to가 실제 존재하는지 확인
        return from_name in all_names and to_name in all_names
    
    def _create_empty_result(self, filename: str) -> Dict[str, Any]:
        """빈 결과 구조 생성"""
        return {
            'source_file': filename,
            'concepts': [],
            'components': [],
            'code_examples': [],
            'tutorials': [],
            'relationships': [],
            'statistics': {
                'concepts': 0,
                'components': 0,
                'code_examples': 0,
                'tutorials': 0,
                'relationships': 0
            }
        }
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        단일 마크다운 파일 처리
        
        Args:
            input_path: 입력 마크다운 파일 경로
            output_path: 출력 JSON 파일 경로 (None이면 자동 생성)
            
        Returns:
            전처리 결과
        """
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")
        
        # 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 파일명 추출 (확장자 제외)
        filename = input_path.stem
        
        # 전처리 수행
        result = self.preprocess_markdown(content, filename)
        
        # 출력 경로 설정
        if output_path is None:
            output_path = input_path.parent / f"{filename}.preprocessed.json"
        
        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 전처리 완료: {output_path}")
        print(f"📊 통계: {result['statistics']}")
        
        return result
    
    def process_batch(self, input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """
        디렉토리 내 모든 마크다운 파일 배치 처리
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            
        Returns:
            모든 파일의 전처리 결과 리스트
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        
        # 출력 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 마크다운 파일 찾기
        md_files = list(input_dir.rglob("*.md"))
        
        if not md_files:
            print("❌ 마크다운 파일을 찾을 수 없습니다.")
            return []
        
        print(f"📁 {len(md_files)}개 파일 발견")
        
        results = []
        for md_file in md_files:
            try:
                # 상대 경로 구조 유지
                relative_path = md_file.relative_to(input_dir)
                output_file = output_dir / relative_path.with_suffix('.preprocessed.json')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                result = self.process_file(md_file, output_file)
                results.append(result)
                
            except Exception as e:
                print(f"❌ {md_file} 처리 실패: {e}")
                continue
        
        # 전체 통계 출력
        self._print_batch_statistics(results)
        
        return results
    
    def _print_batch_statistics(self, results: List[Dict[str, Any]]):
        """배치 처리 통계 출력"""
        if not results:
            return
        
        total_stats = {
            'concepts': 0,
            'components': 0,
            'code_examples': 0,
            'tutorials': 0,
            'relationships': 0
        }
        
        for result in results:
            stats = result.get('statistics', {})
            for key in total_stats:
                total_stats[key] += stats.get(key, 0)
        
        print("\n" + "="*50)
        print("📊 배치 처리 완료 통계")
        print("="*50)
        print(f"처리된 파일 수: {len(results)}")
        for key, value in total_stats.items():
            print(f"{key.title()}: {value}")
        print("="*50)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph 마크다운 파일 Graph DB 전처리")
    parser.add_argument("input", help="입력 파일 또는 디렉토리")
    parser.add_argument("output", nargs='?', help="출력 파일 또는 디렉토리")
    parser.add_argument("--batch", action="store_true", help="배치 처리 모드")
    parser.add_argument("--model", default="gpt-4.1", help="사용할 모델 (기본값: gpt-4.1)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    # 전처리기 초기화
    preprocessor = LLMGraphPreprocessor(model_name=args.model)
    
    try:
        if args.batch:
            # 배치 처리
            if not output_path:
                output_path = input_path.parent / "preprocessed"
            
            results = preprocessor.process_batch(input_path, output_path)
            print(f"\n✅ 배치 처리 완료: {len(results)}개 파일")
            
        else:
            # 단일 파일 처리
            result = preprocessor.process_file(input_path, output_path)
            print(f"\n✅ 파일 처리 완료")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 