# Memory/memory_handler.py
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from neo4j import GraphDatabase
import mem0

@dataclass
class Memory:
    content: str
    timestamp: datetime
    user_id: str
    username: str
    channel_id: str
    mentioned_users: List[str] = field(default_factory=list)
    type: str = "statement"
    confidence: float = 0.8

class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.memory = mem0.Memory()
        self._init_db()
    
    def _init_db(self):
        with self.driver.session() as session:
            # Create Memory constraint
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE
            """)
            # Create User constraint
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE
            """)
    
    async def store_memory(self, memory: Memory):
        # Store in Neo4j
        with self.driver.session() as session:
            session.execute_write(self._create_memory_node, memory)
        
        # Store in mem0
        memory_id = str(hash(f"{memory.timestamp}{memory.content}"))
        self.memory.add(
            id=memory_id,
            content=memory.content,
            metadata={
                "user_id": memory.user_id,
                "username": memory.username,
                "channel_id": memory.channel_id,
                "timestamp": memory.timestamp.isoformat(),
                "type": memory.type
            }
        )

    async def get_relevant_memories(self, query: str, channel_id: str, user_id) -> List[Memory]:
        # Basic search without filters
        results = self.memory.search(query=query, user_id=user_id)
        
        # Apply filters and limits after search
        filtered_results = [
            r for r in results 
            if r.metadata["channel_id"] == channel_id
        ][:5]
        
        memories = []
        with self.driver.session() as session:
            for result in filtered_results:
                memory = session.execute_read(
                    self._get_memory_by_id,
                    result.id
                )
                if memory:
                    memories.append(memory)
        
        return memories

    def _create_memory_node(self, tx, memory: Memory):
        memory_id = str(hash(f"{memory.timestamp}{memory.content}"))
        tx.run("""
        MERGE (m:Memory {
            id: $id,
            content: $content,
            timestamp: $timestamp,
            user_id: $user_id,
            username: $username,
            channel_id: $channel_id,
            type: $type,
            confidence: $confidence
        })
        """, id=memory_id, 
             content=memory.content,
             timestamp=memory.timestamp,
             user_id=memory.user_id,
             username=memory.username,
             channel_id=memory.channel_id,
             type=memory.type,
             confidence=memory.confidence)

        # Create user relationships
        tx.run("""
        MERGE (u:User {id: $user_id, username: $username})
        WITH u
        MATCH (m:Memory {id: $memory_id})
        MERGE (u)-[:SAID]->(m)
        """, user_id=memory.user_id, 
             username=memory.username,
             memory_id=memory_id)

    def _get_memory_by_id(self, tx, memory_id: str) -> Optional[Memory]:
        result = tx.run("""
        MATCH (m:Memory {id: $id})
        RETURN m
        """, id=memory_id)
        
        record = result.single()
        if record:
            data = record["m"]
            return Memory(
                content=data["content"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                user_id=data["user_id"],
                username=data["username"],
                channel_id=data["channel_id"],
                type=data["type"],
                confidence=data["confidence"]
            )
        return None

    def close(self):
        self.driver.close()