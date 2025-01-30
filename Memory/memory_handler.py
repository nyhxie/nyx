from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
from datetime import datetime
import logging

class MemoryHandler:
    def __init__(self, uri, username, password, bot_id):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bot_id = bot_id
        
        # Initialize database schema
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.discord_id IS UNIQUE")
            session.run("CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE")
            
            # Create bot user node
            session.run("""
                MERGE (b:User {discord_id: $bot_id})
                SET b.name = 'AI Bot',
                    b.is_bot = true,
                    b.created_at = datetime()
            """, bot_id=bot_id)

    def close(self):
        self.driver.close()

    def _create_embedding(self, text):
        return self.embedding_model.encode(text).tolist()

    def store_user_message(self, content, user_id, username=None, reply_to_id=None, discord_msg_id=None):
        """Store a user message and create/update relationships"""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (u:User {discord_id: $user_id})
                SET u.last_seen = datetime()
                SET u.name = CASE WHEN $username IS NULL THEN u.name ELSE $username END
                CREATE (m:Message {
                    id: randomUUID(),
                    content: $content,
                    embedding: $embedding,
                    timestamp: datetime(),
                    type: 'user',
                    discord_id: $discord_msg_id
                })
                CREATE (u)-[:SENT]->(m)
                WITH m
                OPTIONAL MATCH (prev:Message {discord_id: $reply_to_id})
                FOREACH(x IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (m)-[:REPLIES_TO]->(prev)
                )
                RETURN m.id as msg_id, m.timestamp as timestamp
            """, 
                user_id=user_id,
                username=username,
                content=content,
                embedding=self._create_embedding(content),
                reply_to_id=reply_to_id,
                discord_msg_id=discord_msg_id
            )
            record = result.single()
            return record if record else None

    def store_bot_response(self, content, reply_to_msg_id, discord_msg_id=None):
        """Store bot's response and link it to the user's message"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (b:User {discord_id: $bot_id})
                MATCH (prev:Message {discord_id: $reply_to_id})
                CREATE (m:Message {
                    id: randomUUID(),
                    content: $content,
                    embedding: $embedding,
                    timestamp: datetime(),
                    type: 'bot',
                    discord_id: $discord_msg_id
                })
                CREATE (b)-[:SENT]->(m)
                CREATE (m)-[:REPLIES_TO]->(prev)
                RETURN m.id as msg_id
            """,
                bot_id=self.bot_id,
                content=content,
                embedding=self._create_embedding(content),
                reply_to_id=reply_to_msg_id,
                discord_msg_id=discord_msg_id
            )
            record = result.single()
            return record["msg_id"] if record else None

    def rebuild_conversation_chain(self, user_id, limit=50):
        """Rebuild the conversation chain between user and bot"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (u:User {discord_id: $user_id})-[:SENT]->(m:Message)
                WITH m
                MATCH path = (m)-[:REPLIES_TO*0..1]-(related:Message)
                WHERE related.timestamp <= m.timestamp
                WITH related
                ORDER BY related.timestamp DESC
                LIMIT $limit
                WITH collect(related) as messages
                UNWIND messages as msg
                MATCH (sender:User)-[:SENT]->(msg)
                RETURN msg.content as content,
                       msg.timestamp as timestamp,
                       msg.type as type,
                       sender.discord_id as author_id,
                       sender.name as author_name
                ORDER BY msg.timestamp ASC
            """, user_id=user_id, limit=limit)
            
            return [{
                "content": r["content"],
                "timestamp": r["timestamp"],
                "role": "assistant" if r["type"] == "bot" else "user",
                "author": {
                    "id": r["author_id"],
                    "name": r["author_name"]
                }
            } for r in results]

    def search_memories(self, query, min_similarity=0.6, limit=5):
        """Search through memories using semantic similarity"""
        query_embedding = self._create_embedding(query)
        
        with self.driver.session() as session:
            results = session.run("""
                MATCH (m:Message)
                WITH m, gds.similarity.cosine($query_embedding, m.embedding) AS score
                WHERE score >= $min_similarity
                MATCH (sender:User)-[:SENT]->(m)
                WITH m, score, sender
                OPTIONAL MATCH (m)<-[:REPLIES_TO]-(response:Message)<-[:SENT]-(bot:User {is_bot: true})
                RETURN m.content as content,
                       m.timestamp as timestamp,
                       m.type as type,
                       score,
                       sender.name as author_name,
                       response.content as response_content
                ORDER BY score DESC
                LIMIT $limit
            """, 
                query_embedding=query_embedding,
                min_similarity=min_similarity,
                limit=limit
            )
            
            return [{
                "content": r["content"],
                "response": r["response_content"],
                "timestamp": r["timestamp"],
                "relevance": r["score"],
                "author": r["author_name"]
            } for r in results]

    def create_or_update_user(self, discord_id, name=None):
        with self.driver.session() as session:
            session.run("""
                MERGE (u:User {discord_id: $discord_id})
                SET u.name = $name
            """, discord_id=discord_id, name=name)

    def update_memory_message_id(self, memory_id, message_id):
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Memory {id: $memory_id})
                SET m.message_id = $message_id
            """, memory_id=memory_id, message_id=message_id)

    def get_user_context(self, user_discord_id, limit=10):
        with self.driver.session() as session:
            results = session.run("""
                MATCH (u:User {discord_id: $user_id})-[:SAID]->(m:Memory)
                RETURN m.content as content, 
                       m.timestamp as timestamp,
                       m.author_id as author_id
                ORDER BY m.timestamp DESC
                LIMIT $limit
            """, user_id=user_discord_id, limit=limit)
            
            return [{
                "content": r["content"],
                "timestamp": r["timestamp"],
                "role": "assistant" if r["author_id"] == self.bot_id else "user"
            } for r in results]

    def get_conversation_context(self, user_discord_id, limit=1000):
        """Get conversation context including both user and bot messages"""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (u:User {discord_id: $user_id})-[:SAID]->(m:Memory)
                WITH m
                MATCH (m)-[:REPLIES_TO*0..1]-(related:Memory)
                WHERE related.timestamp <= m.timestamp
                WITH related
                ORDER BY related.timestamp DESC
                LIMIT $limit
                RETURN related.content as content,
                       related.timestamp as timestamp,
                       related.author_id as author_id
                ORDER BY related.timestamp ASC
            """, user_id=user_discord_id, limit=limit)
            
            return [{
                "content": r["content"],
                "timestamp": r["timestamp"],
                "role": "assistant" if r["author_id"] == self.bot_id else "user"
            } for r in results]

    def get_user_connections(self, user_discord_id):
        with self.driver.session() as session:
            results = session.run("""
                MATCH (u:User {discord_id: $user_id})-[:KNOWS]->(other:User)
                RETURN other.discord_id as discord_id, other.name as name
            """, user_id=user_discord_id)
            
            return [dict(r) for r in results]

    def update_user_known_name(self, discord_id, known_name):
        """Update a user's known name while preserving their Discord username"""
        with self.driver.session() as session:
            session.run("""
                MATCH (u:User {discord_id: $discord_id})
                SET u.known_name = $known_name,
                    u.name_updated_at = datetime()
                RETURN u
            """, discord_id=discord_id, known_name=known_name)
