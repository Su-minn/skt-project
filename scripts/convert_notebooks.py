#!/usr/bin/env python3
"""
ipynb 파일을 markdown으로 변환하는 스크립트

사용법:
    python scripts/convert_notebooks.py [--specific]
    --specific: 지정된 특정 파일들만 변환 (day5: 1,7,8 + day6: 1-6)
"""

import os
import subprocess
from pathlib import Path
import json
import argparse

def convert_notebook_to_markdown(notebook_path: Path, output_dir: Path) -> bool:
    """단일 notebook 파일을 markdown으로 변환"""
    try:
        # 출력 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 출력 파일명 생성 (.ipynb -> .md)
        output_file = output_dir / f"{notebook_path.stem}.md"
        
        # nbconvert 명령어 실행
        cmd = [
            "uv", "run", "jupyter", "nbconvert",
            "--to", "markdown",
            "--output", str(output_file),
            str(notebook_path)
        ]
        
        print(f"변환 중: {notebook_path.name} -> {output_file.name}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ 성공: {output_file}")
            return True
        else:
            print(f"❌ 실패: {notebook_path.name}")
            print(f"에러: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 예외 발생: {notebook_path.name} - {str(e)}")
        return False

def get_specific_files(study_source_dir: Path) -> list:
    """특정 파일들의 경로를 반환"""
    target_files = [
        # day5: 1, 7, 8번
        "day5/DAY05_001_LangGraph_StateGraph.ipynb",
        "day5/DAY05_007_LangGraph_MessageGraph.ipynb", 
        "day5/DAY05_008_LangGraph_ReAct.ipynb",
        # day6: 1-6번
        "day6/DAY06_001_LangGraph_Memory.ipynb",
        "day6/DAY06_002_LangGraph_HITL.ipynb",
        "day6/DAY06_003_LangGraph_SubGraph.ipynb",
        "day6/DAY06_004_LangGraph_Multi-Agent.ipynb",
        "day6/DAY06_005_LangGraph_SelfRAG.ipynb",
        "day6/DAY06_006_LangGraph_CRAG.ipynb"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in target_files:
        full_path = study_source_dir / file_path
        if full_path.exists():
            existing_files.append(full_path)
        else:
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  누락된 파일들:")
        for file in missing_files:
            print(f"  - {file}")
    
    return existing_files

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Jupyter Notebook을 Markdown으로 변환")
    parser.add_argument("--specific", action="store_true", 
                       help="특정 파일들만 변환 (day5: 1,7,8 + day6: 1-6)")
    
    args = parser.parse_args()
    
    project_root = Path.cwd()
    study_source_dir = project_root / "study_source"
    markdown_output_dir = project_root / "data" / "processed" / "markdown"
    
    # study_source 디렉토리 존재 확인
    if not study_source_dir.exists():
        print(f"❌ study_source 디렉토리가 없습니다: {study_source_dir}")
        return
    
    # 변환할 파일들 결정
    if args.specific:
        print("🎯 특정 파일들만 변환합니다 (day5: 1,7,8 + day6: 1-6)")
        notebook_files = get_specific_files(study_source_dir)
        if not notebook_files:
            print("❌ 지정된 파일들을 찾을 수 없습니다.")
            return
    else:
        # 모든 .ipynb 파일 찾기
        notebook_files = list(study_source_dir.rglob("*.ipynb"))
        if not notebook_files:
            print("❌ ipynb 파일을 찾을 수 없습니다.")
            return
    
    print(f"📚 총 {len(notebook_files)}개의 notebook 파일을 변환합니다.")
    
    # 변환할 파일 목록 출력
    print("\n📋 변환 대상 파일들:")
    for notebook_path in sorted(notebook_files):
        relative_path = notebook_path.relative_to(study_source_dir)
        print(f"  - {relative_path}")
    
    # 변환 결과 추적
    success_count = 0
    failed_files = []
    
    print(f"\n🔄 변환 시작...")
    # 각 파일을 변환
    for notebook_path in sorted(notebook_files):
        # 상대 경로 기반으로 출력 디렉토리 구조 유지
        relative_path = notebook_path.relative_to(study_source_dir)
        output_subdir = markdown_output_dir / relative_path.parent
        
        if convert_notebook_to_markdown(notebook_path, output_subdir):
            success_count += 1
        else:
            failed_files.append(notebook_path.name)
    
    # 결과 요약
    print("\n" + "="*50)
    print(f"📊 변환 완료 결과:")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {len(failed_files)}개")
    
    if failed_files:
        print(f"\n❌ 실패한 파일들:")
        for file in failed_files:
            print(f"  - {file}")
    
    print(f"\n📁 변환된 파일 위치: {markdown_output_dir}")

if __name__ == "__main__":
    main() 