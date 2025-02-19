import asyncio
import logging
import discord
from ai_handler import AIHandler
from Memory.memory_handler import MemoryHandler
import yaml
import re
from typing import Optional

logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s %(levelname)s: %(message)s")

def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)

class NyxBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        activity = discord.CustomActivity(name = get_config()["status_message"])
        super().__init__(intents=intents, activity = activity)
        self.config = get_config()
        self.memory = MemoryHandler(
            uri=self.config["neo4j"]["uri"],
            username=self.config["neo4j"]["username"],
            password=self.config["neo4j"]["password"],
            bot_id=self.config["client_id"]
        )
        self.ai = AIHandler(self.config, self.memory)
        
        self.message_queue = asyncio.Queue()
        self.name_patterns = [
            r"(?i)my name is\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"(?i)i(?:'|')?m called\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"(?i)call me\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"(?i)i go by\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)"
        ]
        
    async def setup_hook(self):
        self.loop.create_task(self._process_message_queue())
        
    async def on_ready(self):
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="you"
            ),
            status=discord.Status.online
        )
        logging.info(f"Logged in as {self.user} (ID: {self.user.id})")
        
    async def _process_message_queue(self):
        while True:
            message = await self.message_queue.get()
            try:
                await self._handle_message(message)
            except Exception as e:
                logging.error(f"Error processing message: {e}")
            finally:
                self.message_queue.task_done()
                
    async def _extract_name(self, content: str) -> Optional[str]:
        """Extract name from message if present"""
        for pattern in self.name_patterns:
            if match := re.search(pattern, content):
                name = match.group(1).strip()
                # Updated validation to allow alphanumeric characters
                if 2 <= len(name) <= 32 and all(part.isalnum() for part in name.split()):
                    return name
        return None

    async def _handle_message(self, message: discord.Message):
        if not self._should_process_message(message):
            return
            
        async with message.channel.typing():
            # Clean message content
            cleaned_content = message.content
            if not isinstance(message.channel, discord.DMChannel):
                cleaned_content = cleaned_content.replace(self.user.mention, "").strip()
            
            # Check for name declaration
            if declared_name := await self._extract_name(cleaned_content):
                try:
                    self.memory.update_user_known_name(message.author.id, declared_name)
                    logging.info(f"Updated known name for user {message.author.id} to {declared_name}")
                except Exception as e:
                    logging.error(f"Failed to update user name: {e}")
            
            # Get AI response
            response = await self.ai.get_response(
                user_id=message.author.id,
                message=cleaned_content,
                conversation_id=str(message.channel.id)
            )
            
            try:
                # Store user message
                self.memory.store_user_message(
                    content=cleaned_content,
                    user_id=message.author.id,
                    username=str(message.author),
                    reply_to_id=message.reference.message_id if message.reference else None,
                    discord_msg_id=message.id
                )
                
                # Send and store response
                response_msg = await message.channel.send(response)
                self.memory.store_bot_response(
                    content=response,
                    reply_to_msg_id=message.id,  
                    discord_msg_id=response_msg.id
                )
                
            except Exception as e:
                logging.error(f"Error storing message in memory: {e}")
                # Still send response even if storage fails
                if not response_msg:
                    await message.channel.send(response)
    
    def _should_process_message(self, message: discord.Message) -> bool:
        if message.author == self.user:
            return False
            
        is_dm = isinstance(message.channel, discord.DMChannel)
        
        # Check channel and user permissions
        if not is_dm and self.user not in message.mentions:
            return False
            
        if not self.config["allow_dms"] and is_dm:
            return False
            
        allowed_channels = self.config["allowed_channel_ids"]
        if allowed_channels and message.channel.id not in allowed_channels:
            return False
            
        if message.author.id in self.config["blocked_user_ids"]:
            return False
            
        return True
        
    async def on_message(self, message: discord.Message):
        await self.message_queue.put(message)
        
    async def on_shutdown(self):
        self.memory.close()

def main():
    bot = NyxBot()
    bot.run(bot.config["bot_token"])

if __name__ == "__main__":
    main()