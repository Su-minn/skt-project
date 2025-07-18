#!/usr/bin/env python3
"""
JSON 전처리 결과 → Neo4j 저장 스크립트

기능:
- 전처리된 JSON 파일에서 LangGraph 학습 데이터 추출
- Neo4j에 노드(Concept, Component, CodeExample, Tutorial) 및 관계 저장
- 임베딩 벡터 생성 및 저장으로 RAG 검색 지원
- 배치 처리 지원

사용법:
    # 단일 파일 처리
    python scripts/json_to_neo4j_parser.py data/processed/json/day5/DAY05_001_LangGraph_StateGraph.preprocessed.json
    
    # 배치 처리 (디렉토리 내 모든 JSON 파일)
    python scripts/json_to_neo4j_parser.py data/processed/json --batch
"""

import sys
import json
import uuid
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class GraphNode:
    """Graph DB 노드 데이터 클래스"""
    node_type: str  # "Concept", "Component", "CodeExample", "Tutorial"
    name: str       # name 또는 title
    description: str
    source: str
    code: Optional[str] = None  # CodeExample, Tutorial만 해당
    node_id: str = None
    
    def __post_init__(self):
        if self.node_id is None:
            self.node_id = str(uuid.uuid4())

@dataclass
class GraphRelationship:
    """Graph DB 관계 데이터 클래스"""
    from_node: str
    to_node: str
    relationship_type: str
    weight: float = 1.0

class JsonToNeo4jParser:
    """JSON을 Neo4j로 저장하는 파서"""
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username, 
            password=neo4j_password
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 통계 추적
        self.stats = {
            "concepts": 0,
            "components": 0,  
            "code_examples": 0,
            "tutorials": 0,
            "relationships": 0,
            "files_processed": 0
        }
    
    def setup_neo4j_schema(self):
        """Neo4j 스키마 설정"""
        print("🏗️ Neo4j 스키마 설정 중...")
        
        # 제약조건 생성
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (comp:Component) REQUIRE comp.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ce:CodeExample) REQUIRE ce.title IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tutorial) REQUIRE t.title IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.node_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self.graph.query(constraint)
            except Exception as e:
                print(f"⚠️ 제약조건 생성 중 경고: {e}")
        
        # 벡터 인덱스 생성
        vector_indexes = [
            """
            CREATE VECTOR INDEX concept_embedding_index IF NOT EXISTS
            FOR (c:Concept) 
            ON (c.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            """
            CREATE VECTOR INDEX component_embedding_index IF NOT EXISTS
            FOR (comp:Component) 
            ON (comp.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            """
            CREATE VECTOR INDEX code_example_embedding_index IF NOT EXISTS
            FOR (ce:CodeExample) 
            ON (ce.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """,
            """
            CREATE VECTOR INDEX tutorial_embedding_index IF NOT EXISTS
            FOR (t:Tutorial) 
            ON (t.embedding)
            OPTIONS {
              indexConfig: {
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
              }
            }
            """
        ]
        
        for index in vector_indexes:
            try:
                self.graph.query(index)
            except Exception as e:
                print(f"⚠️ 벡터 인덱스 생성 중 경고: {e}")
        
        print("✅ Neo4j 스키마 설정 완료")
    
    def load_json_data(self, json_file: Path) -> Dict[str, Any]:
        """JSON 파일 로드"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ JSON 파일 로드 실패: {json_file} - {e}")
            return {}
    
    def create_embedding_text(self, node: GraphNode) -> str:
        """임베딩을 위한 텍스트 생성"""
        text_parts = [node.name, node.description]
        
        if node.code:
            # 코드가 있는 경우 코드의 일부도 포함 (너무 길면 잘라냄)
            code_snippet = node.code[:500] if len(node.code) > 500 else node.code
            text_parts.append(code_snippet)
        
        return " ".join(filter(None, text_parts))
    
    def create_node_in_neo4j(self, node: GraphNode) -> bool:
        """Neo4j에 노드 생성"""
        try:
            # 임베딩 생성
            embedding_text = self.create_embedding_text(node)
            embedding = self.embeddings.embed_query(embedding_text)
            
            # 노드 타입에 따른 쿼리 구성
            if node.node_type in ["Concept", "Component"]:
                # name을 사용하는 노드들
                query = f"""
                MERGE (n:{node.node_type} {{name: $name}})
                SET n.description = $description,
                    n.source = $source,
                    n.node_id = $node_id,
                    n.created_at = datetime(),
                    n.updated_at = datetime()
                WITH n
                CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                RETURN n.node_id as node_id
                """
                
                params = {
                    "name": node.name,
                    "description": node.description,
                    "source": node.source,
                    "node_id": node.node_id,
                    "embedding": embedding
                }
                
            else:  # CodeExample, Tutorial
                # title을 사용하는 노드들
                query = f"""
                MERGE (n:{node.node_type} {{title: $title}})
                SET n.description = $description,
                    n.code = $code,
                    n.source = $source,
                    n.node_id = $node_id,
                    n.created_at = datetime(),
                    n.updated_at = datetime()
                WITH n
                CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding)
                RETURN n.node_id as node_id
                """
                
                params = {
                    "title": node.name,  # JSON에서는 title 필드
                    "description": node.description,
                    "code": node.code or "",
                    "source": node.source,
                    "node_id": node.node_id,
                    "embedding": embedding
                }
            
            result = self.graph.query(query, params=params)
            return True
            
        except Exception as e:
            print(f"❌ 노드 생성 실패: {node.name} ({node.node_type}) - {e}")
            return False
    
    def create_relationship_in_neo4j(self, relationship: GraphRelationship) -> bool:
        """Neo4j에 관계 생성"""
        try:
            # 관계 타입을 Neo4j 형식으로 변환
            rel_type = relationship.relationship_type.upper().replace(" ", "_")
            
            query = f"""
            MATCH (from) WHERE from.name = $from_name OR from.title = $from_name
            MATCH (to) WHERE to.name = $to_name OR to.title = $to_name
            MERGE (from)-[r:{rel_type}]->(to)
            SET r.weight = $weight,
                r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "from_name": relationship.from_node,
                "to_name": relationship.to_node,
                "weight": relationship.weight
            }
            
            result = self.graph.query(query, params=params)
            return len(result) > 0
            
        except Exception as e:
            print(f"❌ 관계 생성 실패: {relationship.from_node} -> {relationship.to_node} - {e}")
            return False
    
    def process_json_file(self, json_file: Path) -> bool:
        """단일 JSON 파일 처리"""
        print(f"📄 처리 중: {json_file.name}")
        
        # JSON 데이터 로드
        data = self.load_json_data(json_file)
        if not data:
            return False
        
        # 노드들 생성
        created_nodes = []
        
        # Concepts 처리
        for concept_data in data.get("concepts", []):
            node = GraphNode(
                node_type="Concept",
                name=concept_data.get("name", ""),
                description=concept_data.get("description", ""),
                source=concept_data.get("source", "")
            )
            
            if self.create_node_in_neo4j(node):
                created_nodes.append(node.name)
                self.stats["concepts"] += 1
        
        # Components 처리
        for component_data in data.get("components", []):
            node = GraphNode(
                node_type="Component",
                name=component_data.get("name", ""),
                description=component_data.get("description", ""),
                source=component_data.get("source", "")
            )
            
            if self.create_node_in_neo4j(node):
                created_nodes.append(node.name)
                self.stats["components"] += 1
        
        # CodeExamples 처리
        for code_data in data.get("code_examples", []):
            node = GraphNode(
                node_type="CodeExample",
                name=code_data.get("title", ""),
                description=code_data.get("description", ""),
                source=code_data.get("source", ""),
                code=code_data.get("code", "")
            )
            
            if self.create_node_in_neo4j(node):
                created_nodes.append(node.name)
                self.stats["code_examples"] += 1
        
        # Tutorials 처리
        for tutorial_data in data.get("tutorials", []):
            node = GraphNode(
                node_type="Tutorial",
                name=tutorial_data.get("title", ""),
                description=tutorial_data.get("description", ""),
                source=tutorial_data.get("source", ""),
                code=tutorial_data.get("code", "")
            )
            
            if self.create_node_in_neo4j(node):
                created_nodes.append(node.name)
                self.stats["tutorials"] += 1
        
        # Relationships 처리
        for rel_data in data.get("relationships", []):
            relationship = GraphRelationship(
                from_node=rel_data.get("from", ""),
                to_node=rel_data.get("to", ""),
                relationship_type=rel_data.get("type", "RELATES_TO")
            )
            
            if self.create_relationship_in_neo4j(relationship):
                self.stats["relationships"] += 1
        
        self.stats["files_processed"] += 1
        print(f"✅ 완료: {json_file.name} ({len(created_nodes)}개 노드 생성)")
        return True
    
    def process_batch(self, json_dir: Path) -> List[Path]:
        """배치 처리: 디렉토리 내 모든 JSON 파일 처리"""
        json_files = list(json_dir.rglob("*.json"))
        # 작은 파일들(실패한 파일들) 제외
        json_files = [f for f in json_files if f.stat().st_size > 1024]  # 1KB 이상만
        
        print(f"🗂️ 배치 처리 시작: {len(json_files)}개 파일")
        
        # 스키마 설정
        self.setup_neo4j_schema()
        
        processed_files = []
        for json_file in json_files:
            if self.process_json_file(json_file):
                processed_files.append(json_file)
        
        return processed_files
    
    def print_final_stats(self):
        """최종 통계 출력"""
        print("\n" + "="*60)
        print("📊 Neo4j 저장 완료 통계")
        print("="*60)
        print(f"처리된 파일 수: {self.stats['files_processed']}")
        print(f"Concepts: {self.stats['concepts']}")
        print(f"Components: {self.stats['components']}")
        print(f"CodeExamples: {self.stats['code_examples']}")
        print(f"Tutorials: {self.stats['tutorials']}")
        print(f"Relationships: {self.stats['relationships']}")
        print(f"총 노드 수: {self.stats['concepts'] + self.stats['components'] + self.stats['code_examples'] + self.stats['tutorials']}")
        print("="*60)

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="JSON을 Neo4j Graph DB에 저장")
    parser.add_argument("input", help="입력 JSON 파일 또는 디렉토리")
    parser.add_argument("--batch", action="store_true", help="배치 처리 모드")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ 입력 경로가 없습니다: {input_path}")
        sys.exit(1)
    
    # Neo4j 연결 정보
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME") 
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("❌ Neo4j 연결 정보가 필요합니다 (.env 파일 확인)")
        print("   NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD 설정 필요")
        sys.exit(1)
    
    try:
        # 파서 생성
        parser = JsonToNeo4jParser(neo4j_uri, neo4j_username, neo4j_password)
        
        if args.batch:
            # 배치 처리
            if not input_path.is_dir():
                print("❌ 배치 모드는 디렉토리가 필요합니다")
                sys.exit(1)
            
            processed_files = parser.process_batch(input_path)
            
        else:
            # 단일 파일 처리
            if not input_path.is_file():
                print("❌ 단일 파일 모드는 JSON 파일이 필요합니다")
                sys.exit(1)
            
            parser.setup_neo4j_schema()
            success = parser.process_json_file(input_path)
            processed_files = [input_path] if success else []
        
        # 최종 통계 출력
        parser.print_final_stats()
        
        if processed_files:
            print(f"\n✅ Neo4j 저장 완료!")
            print(f"   - URI: {neo4j_uri}")
            print(f"   - 처리된 파일: {len(processed_files)}개")
        else:
            print("\n❌ 저장된 데이터가 없습니다")
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 