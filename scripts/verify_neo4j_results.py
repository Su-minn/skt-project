#!/usr/bin/env python3
"""
Neo4j Graph DB 저장 결과 검증 스크립트

기능:
- 노드 타입별 개수 확인
- 관계 타입별 개수 확인  
- 샘플 관계 예시 출력
"""

import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

def verify_neo4j_data():
    """Neo4j 저장 결과 검증"""
    
    # Neo4j 연결
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME") 
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username, 
        password=neo4j_password
    )
    
    print("🔍 Neo4j Graph DB 저장 결과 검증")
    print("="*50)
    
    # 1. 노드 타입별 개수
    print("\n📊 노드 타입별 개수:")
    node_counts = graph.query("""
        MATCH (n) 
        RETURN labels(n)[0] as nodeType, count(n) as count
        ORDER BY count DESC
    """)
    
    total_nodes = 0
    for result in node_counts:
        print(f"  {result['nodeType']}: {result['count']}개")
        total_nodes += result['count']
    print(f"  총 노드: {total_nodes}개")
    
    # 2. 관계 타입별 개수
    print("\n🔗 관계 타입별 개수:")
    rel_counts = graph.query("""
        MATCH ()-[r]->()
        RETURN type(r) as relType, count(r) as count
        ORDER BY count DESC
    """)
    
    total_rels = 0
    for result in rel_counts:
        print(f"  {result['relType']}: {result['count']}개")
        total_rels += result['count']
    print(f"  총 관계: {total_rels}개")
    
    # 3. 각 관계 타입별 샘플 예시
    print("\n💡 관계 타입별 샘플 예시:")
    
    relationship_types = ['IMPLEMENTS', 'USES', 'APPLIES', 'INCLUDES', 'REQUIRES']
    
    for rel_type in relationship_types:
        samples = graph.query(f"""
            MATCH (from)-[r:{rel_type}]->(to)
            RETURN from.name as fromName, from.title as fromTitle, 
                   to.name as toName, to.title as toTitle
            LIMIT 2
        """)
        
        print(f"\n  {rel_type}:")
        for sample in samples:
            from_name = sample['fromName'] or sample['fromTitle'] or 'Unknown'
            to_name = sample['toName'] or sample['toTitle'] or 'Unknown'
            print(f"    - {from_name} --{rel_type}--> {to_name}")
    
    # 4. Tutorial과 CodeExample 분류 확인
    print("\n🏷️ Tutorial vs CodeExample 분류:")
    tutorial_samples = graph.query("""
        MATCH (t:Tutorial)
        RETURN t.title as title
        LIMIT 3
    """)
    print("  Tutorial 예시:")
    for sample in tutorial_samples:
        print(f"    - {sample['title']}")
    
    code_samples = graph.query("""
        MATCH (c:CodeExample)
        RETURN c.title as title
        LIMIT 3
    """)
    print("  CodeExample 예시:")
    for sample in code_samples:
        print(f"    - {sample['title']}")
    
    print(f"\n✅ 검증 완료! Neo4j URI: {neo4j_uri}")

if __name__ == "__main__":
    verify_neo4j_data() 