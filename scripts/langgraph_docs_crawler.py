#!/usr/bin/env python3
"""
LangGraph 공식문서 크롤링 스크립트

LangGraph 가이드 페이지들을 크롤링하여 하이브리드 검색용 데이터를 수집합니다.
"""

import os
import time
import json
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangGraphDocsCrawler:
    """LangGraph 공식문서 크롤러"""
    
    def __init__(self, base_url: str = "https://langchain-ai.github.io/langgraph/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # 출력 디렉토리 설정
        self.output_dir = Path("data/langgraph_docs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_page_content(self, url: str) -> Dict[str, Any]:
        """단일 페이지 내용 크롤링"""
        try:
            logger.info(f"크롤링 시작: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 메인 콘텐츠 추출 (Material MkDocs 구조 기반)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='md-content')
            
            if not main_content:
                logger.warning(f"메인 콘텐츠를 찾을 수 없음: {url}")
                return None
                
            # 제목 추출
            title = ""
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # 텍스트 내용 추출 (코드 블록 포함)
            content = ""
            for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code']):
                text = elem.get_text().strip()
                if text:
                    if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        content += f"\n\n## {text}\n\n"
                    elif elem.name in ['pre', 'code']:
                        content += f"\n```\n{text}\n```\n"
                    else:
                        content += f"{text}\n\n"
            
            # 메타데이터 구성
            parsed_url = urlparse(url)
            
            doc_data = {
                "title": title,
                "content": content.strip(),
                "url": url,
                "path": parsed_url.path,
                "domain": parsed_url.netloc,
                "source": "LangGraph 공식문서",
                "content_length": len(content),
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"크롤링 완료: {title} ({len(content)}자)")
            return doc_data
            
        except Exception as e:
            logger.error(f"크롤링 실패 {url}: {e}")
            return None
    
    def crawl_guide_pages(self, max_pages: int = 10) -> List[Dict[str, Any]]:
        """가이드 페이지들 크롤링"""
        
        # 크롤링할 핵심 가이드 페이지들 (작게 시작)
        guide_urls = [
            "https://langchain-ai.github.io/langgraph/guides/",
            "https://langchain-ai.github.io/langgraph/how-tos/",
            "https://langchain-ai.github.io/langgraph/concepts/",
            "https://langchain-ai.github.io/langgraph/tutorials/",
            # 핵심 가이드들
            "https://langchain-ai.github.io/langgraph/concepts/low_level/",
            "https://langchain-ai.github.io/langgraph/concepts/persistence/",
            "https://langchain-ai.github.io/langgraph/concepts/streaming/",
            "https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/",
            "https://langchain-ai.github.io/langgraph/how-tos/state-model/",
            "https://langchain-ai.github.io/langgraph/how-tos/persistence/"
        ]
        
        docs = []
        
        for i, url in enumerate(guide_urls[:max_pages]):
            if i > 0:
                time.sleep(1)  # 서버 부하 방지
                
            doc_data = self.get_page_content(url)
            if doc_data and len(doc_data['content']) > 100:  # 최소 콘텐츠 길이 확인
                docs.append(doc_data)
            
        logger.info(f"총 {len(docs)}개 페이지 크롤링 완료")
        return docs
    
    def save_docs(self, docs: List[Dict[str, Any]]) -> str:
        """크롤링된 문서들을 파일로 저장"""
        
        # JSON Lines 형식으로 저장
        jsonl_file = self.output_dir / "langgraph_docs.jsonl"
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # 요약 정보 저장
        summary = {
            "total_docs": len(docs),
            "total_content_length": sum(len(doc['content']) for doc in docs),
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "LangGraph 공식문서"
        }
        
        summary_file = self.output_dir / "crawl_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"문서 저장 완료: {jsonl_file}")
        logger.info(f"요약 정보: {len(docs)}개 문서, 총 {summary['total_content_length']:,}자")
        
        return str(jsonl_file)


def main():
    """메인 실행 함수"""
    print("🚀 LangGraph 공식문서 크롤링 시작")
    print("=" * 50)
    
    # 크롤러 초기화
    crawler = LangGraphDocsCrawler()
    
    # 가이드 페이지들 크롤링 (10개 페이지로 시작)
    docs = crawler.crawl_guide_pages(max_pages=10)
    
    if not docs:
        print("❌ 크롤링된 문서가 없습니다.")
        return
    
    # 결과 저장
    saved_file = crawler.save_docs(docs)
    
    print("\n✅ 크롤링 완료!")
    print(f"📄 수집된 문서: {len(docs)}개")
    print(f"💾 저장 위치: {saved_file}")
    
    # 샘플 문서 출력
    if docs:
        sample = docs[0]
        print(f"\n📖 샘플 문서:")
        print(f"제목: {sample['title']}")
        print(f"URL: {sample['url']}")
        print(f"내용 길이: {len(sample['content'])}자")
        print(f"내용 미리보기: {sample['content'][:200]}...")


if __name__ == "__main__":
    main() 