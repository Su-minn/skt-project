#!/usr/bin/env python3
"""
마크다운 → Neo4j 파싱 및 저장 스크립트

기능:
- 정제된 마크다운에서 LangGraph 개념 추출
- 견고한 파싱: 구조 변화에 대응하는 유연한 로직
- Neo4j 노드/관계 생성 및 임베딩 저장
- RAG 검색을 위한 메타데이터 구조화

사용법:
    python scripts/markdown_to_neo4j_parser.py <input_file>
"""

import sys
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class ConceptNode:
    """LangGraph 개념 노드 데이터 클래스"""
    name: str
    node_type: str  # "concept", "tutorial", "pattern", "tool", "usecase"
    definition: str = ""
    characteristics: List[str] = field(default_factory=list)
    usage: str = ""
    code_examples: List[str] = field(default_factory=list)
    subsections: Dict[str, str] = field(default_factory=dict)
    source_file: str = ""
    section_order: int = 0
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ConceptRelationship:
    """개념 간 관계 데이터 클래스"""
    from_concept: str
    to_concept: str
    relationship_type: str  # "PREREQUISITE", "BUILDS_UPON", "IMPLEMENTS", "TEACHES", "SUPPORTS", "RELATES_TO"
    weight: float = 1.0

class MarkdownToNeo4jParser:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username, 
            password=neo4j_password
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 파싱 패턴 정의
        self.concept_pattern = re.compile(r'^## (.+?)$', re.MULTILINE)
        self.definition_pattern = re.compile(r'\*\*정의\*\*:\s*(.+?)(?=\n\n|\*\*|###|##|$)', re.DOTALL)
        self.characteristics_pattern = re.compile(r'\*\*특징\*\*:\s*\n\n((?:- .+?\n)+)', re.DOTALL)
        self.usage_pattern = re.compile(r'\*\*활용\*\*:\s*(.+?)(?=\n\n|\*\*|###|##|$)', re.DOTALL)
        self.code_block_pattern = re.compile(r'```[\w]*\n(.*?)\n```', re.DOTALL)
        
        # 관계 패턴들 (유연한 매칭)
        self.relationship_patterns = {
            'prerequisite': re.compile(r'\*\*선행 개념\*\*:\s*\[(.+?)\]', re.DOTALL),
            'related': re.compile(r'\*\*연관 개념\*\*:\s*\[(.+?)\]', re.DOTALL),
            'opposite': re.compile(r'\*\*반대 개념\*\*:\s*\[(.+?)\]', re.DOTALL),
        }
    
    def setup_neo4j_schema(self):
        """Neo4j 스키마 설정"""
        print("🏗️ Neo4j 스키마 설정 중...")
        
        # 제약조건 생성
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tutorial) REQUIRE t.title IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE"
        ]
        
        for constraint in constraints:
            self.graph.query(constraint)
        
        # 벡터 인덱스 생성
        vector_index = """
        CREATE VECTOR INDEX concept_embedding_index IF NOT EXISTS
        FOR (c:Concept) 
        ON (c.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
          }
        }
        """
        self.graph.query(vector_index)
        
        # 추가 벡터 인덱스들
        additional_indexes = [
            """
            CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
            FOR (ch:Chunk) 
            ON (ch.embedding)
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
        
        for index in additional_indexes:
            self.graph.query(index)
        
        print("✅ Neo4j 스키마 설정 완료")
    
    def clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text:
            return ""
        
        # 불필요한 공백과 줄바꿈 제거
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_concept_sections(self, content: str) -> List[Tuple[str, str]]:
        """개념 섹션들을 추출"""
        sections = []
        concept_matches = list(self.concept_pattern.finditer(content))
        
        for i, match in enumerate(concept_matches):
            concept_name = match.group(1).strip()
            start_pos = match.end()
            
            # 다음 섹션의 시작 위치 찾기
            if i + 1 < len(concept_matches):
                end_pos = concept_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            section_content = content[start_pos:end_pos].strip()
            sections.append((concept_name, section_content))
        
        return sections
    
    def determine_node_type(self, concept_name: str, content: str) -> str:
        """노드 타입 결정"""
        concept_lower = concept_name.lower()
        content_lower = content.lower()
        
        if '[실습]' in concept_name or 'tutorial' in concept_lower:
            return "tutorial"
        elif '실행' in concept_name or 'stream' in concept_lower or 'invoke' in concept_lower:
            return "pattern"
        elif 'studio' in concept_lower or 'tool' in concept_lower:
            return "tool"
        elif 'vs.' in concept_name or '비교' in concept_name:
            return "usecase"
        else:
            return "concept"
    
    def parse_concept_content(self, concept_name: str, content: str, source_file: str, section_order: int) -> ConceptNode:
        """개념 내용 파싱 (견고한 에러 처리)"""
        node_type = self.determine_node_type(concept_name, content)
        concept = ConceptNode(
            name=concept_name,
            node_type=node_type,
            source_file=source_file,
            section_order=section_order
        )
        
        try:
            # 정의 추출
            def_match = self.definition_pattern.search(content)
            if def_match:
                concept.definition = self.clean_text(def_match.group(1))
            
            # 특징 추출
            char_match = self.characteristics_pattern.search(content)
            if char_match:
                characteristics_text = char_match.group(1)
                characteristics = re.findall(r'- (.+?)(?=\n|$)', characteristics_text)
                concept.characteristics = [self.clean_text(c) for c in characteristics]
            
            # 활용 추출
            usage_match = self.usage_pattern.search(content)
            if usage_match:
                concept.usage = self.clean_text(usage_match.group(1))
            
            # 코드 예제 추출
            code_matches = self.code_block_pattern.findall(content)
            concept.code_examples = [code.strip() for code in code_matches if code.strip()]
            
            # 하위 섹션 추출
            subsection_pattern = re.compile(r'### (.+?)\n(.*?)(?=###|##|$)', re.DOTALL)
            subsection_matches = subsection_pattern.findall(content)
            for subsection_title, subsection_content in subsection_matches:
                concept.subsections[subsection_title.strip()] = self.clean_text(subsection_content)
        
        except Exception as e:
            print(f"⚠️ 개념 '{concept_name}' 파싱 중 에러: {e}")
            # 에러가 발생해도 기본 정보는 유지
        
        return concept
    
    def extract_relationships(self, content: str) -> List[ConceptRelationship]:
        """관계 정보 추출 (유연한 패턴 매칭)"""
        relationships = []
        
        # 관계 섹션 찾기
        relationship_section_pattern = re.compile(r'\*\*관계\*\*:\s*(.*?)(?=\n---|\n##|$)', re.DOTALL)
        rel_match = relationship_section_pattern.search(content)
        
        if not rel_match:
            return relationships
        
        relationship_content = rel_match.group(1)
        
        # 각 관계 타입별로 추출
        for rel_type, pattern in self.relationship_patterns.items():
            matches = pattern.findall(relationship_content)
            for match in matches:
                # 개념들을 분리 (쉼표, 괄호 등 고려)
                concepts = re.split(r'[,，]|\s+', match)
                concepts = [self.clean_concept_name(c) for c in concepts if c.strip()]
                
                for concept in concepts:
                    if concept:
                        relationships.append(ConceptRelationship(
                            from_concept="",  # 나중에 설정
                            to_concept=concept,
                            relationship_type=self.map_relationship_type(rel_type)
                        ))
        
        return relationships
    
    def clean_concept_name(self, name: str) -> str:
        """개념 이름 정제"""
        # 대괄호, 괄호 제거
        name = re.sub(r'[\[\]()]', '', name)
        # 불필요한 공백 제거
        name = name.strip()
        return name
    
    def map_relationship_type(self, rel_type: str) -> str:
        """관계 타입 매핑"""
        mapping = {
            'prerequisite': 'PREREQUISITE',
            'related': 'RELATES_TO', 
            'opposite': 'OPPOSITE_OF'
        }
        return mapping.get(rel_type, 'RELATES_TO')
    
    def create_concept_node(self, concept: ConceptNode) -> str:
        """Neo4j에 개념 노드 생성"""
        # 임베딩 생성 (정의 + 특징 + 활용)
        embedding_text = f"{concept.definition} {' '.join(concept.characteristics)} {concept.usage}"
        embedding = self.embeddings.embed_query(embedding_text)
        
        # 노드 생성 쿼리
        query = f"""
        MERGE (c:{concept.node_type.title()} {{name: $name}})
        SET c.definition = $definition,
            c.characteristics = $characteristics,
            c.usage = $usage,
            c.code_examples = $code_examples,
            c.source_file = $source_file,
            c.section_order = $section_order,
            c.chunk_id = $chunk_id,
            c.node_type = $node_type,
            c.created_at = datetime()
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
        RETURN c.chunk_id as chunk_id
        """
        
        params = {
            "name": concept.name,
            "definition": concept.definition,
            "characteristics": concept.characteristics,
            "usage": concept.usage,
            "code_examples": concept.code_examples,
            "source_file": concept.source_file,
            "section_order": concept.section_order,
            "chunk_id": concept.chunk_id,
            "node_type": concept.node_type,
            "embedding": embedding
        }
        
        result = self.graph.query(query, params=params)
        return result[0]["chunk_id"] if result else concept.chunk_id
    
    def create_subsection_chunks(self, concept: ConceptNode):
        """하위 섹션을 별도 청크로 생성"""
        for subsection_title, subsection_content in concept.subsections.items():
            if not subsection_content.strip():
                continue
                
            chunk_id = str(uuid.uuid4())
            embedding = self.embeddings.embed_query(subsection_content)
            
            query = """
            MATCH (c {name: $concept_name})
            CREATE (ch:Chunk {
                chunk_id: $chunk_id,
                title: $title,
                content: $content,
                parent_concept: $concept_name,
                source_file: $source_file,
                chunk_type: 'subsection',
                created_at: datetime()
            })
            WITH ch, c
            CALL db.create.setNodeVectorProperty(ch, 'embedding', $embedding)
            CREATE (c)-[:HAS_SUBSECTION]->(ch)
            RETURN ch.chunk_id
            """
            
            params = {
                "concept_name": concept.name,
                "chunk_id": chunk_id,
                "title": subsection_title,
                "content": subsection_content,
                "source_file": concept.source_file,
                "embedding": embedding
            }
            
            self.graph.query(query, params=params)
    
    def create_relationships(self, relationships: List[ConceptRelationship]):
        """관계 생성"""
        for rel in relationships:
            if not rel.from_concept or not rel.to_concept:
                continue
                
            query = f"""
            MATCH (from {{name: $from_concept}})
            MATCH (to {{name: $to_concept}})
            MERGE (from)-[r:{rel.relationship_type}]->(to)
            SET r.weight = $weight,
                r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "from_concept": rel.from_concept,
                "to_concept": rel.to_concept,
                "weight": rel.weight
            }
            
            try:
                self.graph.query(query, params=params)
            except Exception as e:
                print(f"⚠️ 관계 생성 실패: {rel.from_concept} -> {rel.to_concept} ({e})")
    
    def parse_and_store(self, markdown_file: str):
        """마크다운 파일 파싱 및 Neo4j 저장"""
        print(f"📄 파일 파싱 시작: {markdown_file}")
        
        # 파일 읽기
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 개념 섹션 추출
        concept_sections = self.extract_concept_sections(content)
        print(f"📊 추출된 개념 섹션: {len(concept_sections)}개")
        
        # 스키마 설정
        self.setup_neo4j_schema()
        
        # 개념들과 관계들 저장
        all_relationships = []
        created_concepts = []
        
        for i, (concept_name, section_content) in enumerate(concept_sections):
            try:
                print(f"🔄 처리 중: {concept_name}")
                
                # 개념 파싱
                concept = self.parse_concept_content(
                    concept_name, 
                    section_content, 
                    Path(markdown_file).name, 
                    i
                )
                
                # 노드 생성
                chunk_id = self.create_concept_node(concept)
                created_concepts.append(concept.name)
                
                # 하위 섹션 청크 생성
                if concept.subsections:
                    self.create_subsection_chunks(concept)
                
                # 관계 추출
                relationships = self.extract_relationships(section_content)
                for rel in relationships:
                    rel.from_concept = concept_name
                    all_relationships.append(rel)
                
                print(f"✅ 완료: {concept_name} (청크ID: {chunk_id[:8]}...)")
                
            except Exception as e:
                print(f"❌ 에러: {concept_name} - {e}")
        
        # 관계 생성
        print(f"\n🔗 관계 생성 중: {len(all_relationships)}개")
        self.create_relationships(all_relationships)
        
        # 요약 출력
        print(f"\n🎉 파싱 완료!")
        print(f"   - 생성된 개념: {len(created_concepts)}개")
        print(f"   - 처리된 관계: {len(all_relationships)}개")
        print(f"   - 소스 파일: {Path(markdown_file).name}")
        
        return created_concepts, all_relationships

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("사용법: python scripts/markdown_to_neo4j_parser.py <input_file>")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if not input_file.exists():
        print(f"❌ 입력 파일이 없습니다: {input_file}")
        sys.exit(1)
    
    # Neo4j 연결 정보
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME") 
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        print("❌ Neo4j 연결 정보가 필요합니다 (.env 파일 확인)")
        sys.exit(1)
    
    try:
        # 파서 생성 및 실행
        parser = MarkdownToNeo4jParser(neo4j_uri, neo4j_username, neo4j_password)
        concepts, relationships = parser.parse_and_store(str(input_file))
        
        print(f"\n✅ Neo4j 저장 완료!")
        print(f"   - URI: {neo4j_uri}")
        print(f"   - 개념들: {', '.join(concepts[:5])}{'...' if len(concepts) > 5 else ''}")
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 