from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from Memory.memory_handler import MemoryHandler

class AIHandler:
    def __init__(self, config: dict, memory_handler: MemoryHandler):
        self.config = config
        self.memory = memory_handler
        self.client = httpx.AsyncClient(timeout=3000.0)
        
        # Cache for conversation contexts
        self.conversation_contexts = {}
        self.context_retrieved = set()  # Track channels where context was retrieved
        self.first_message_sent = set()  # Track channels where first message was sent
        
    async def get_response(self, user_id: int, message: str, conversation_id: str) -> str:
        # Initialize or rebuild conversation context if needed
        if conversation_id not in self.context_retrieved:
            self.context_retrieved.add(conversation_id)
            conversation_history = self.memory.rebuild_conversation_chain(user_id)
            self.conversation_contexts[conversation_id] = [
                {"content": msg["content"], "role": msg["role"]}
                for msg in reversed(conversation_history)
            ]
        
        # Get conversation context
        context = self._get_context(conversation_id)
        
        # Build messages array
        messages = self._build_message_array(context, user_id, message)
        
        # Prepare API request
        api_url = f"{self.config['providers']['lmstudio']['base_url']}/chat/completions"
        
        # Only include tools if this isn't the first message in the channel
        is_first_message = conversation_id not in self.first_message_sent
        
        while True:  # Handle multi-turn tool calls
            try:
                request_body = {
                    "messages": messages,
                    "model": self.config["model"],
                    **self.config["extra_api_parameters"]
                }
                
                # Add tools only if not first message
                if not is_first_message:
                    request_body["tools"] = self._get_available_tools()
                    request_body["tool_choice"] = "auto"
                
                response = await self.client.post(api_url, json=request_body)
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"]["content"]
                
                # After successful response, mark channel as having first message
                if is_first_message:
                    self.first_message_sent.add(conversation_id)
                
                # Update context
                self._update_context(conversation_id, message, "user")
                self._update_context(conversation_id, content, "assistant")
                
                # Handle potential tool calls only if not first message
                if not is_first_message and (tool_calls := result["choices"][0]["message"].get("tool_calls")):
                    tool_results = await self._handle_tool_calls(tool_calls)
                    messages.extend(tool_results)
                    continue
                
                return content
                
            except Exception as e:
                logging.error(f"Error in AI response generation: {e}")
                return "Sorry, I encountered an error while processing your message."
    
    def _get_context(self, conversation_id: str) -> List[Dict[str, str]]:
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = []
        return self.conversation_contexts[conversation_id]
    
    def _update_context(self, conversation_id: str, content: str, role: str):
        context = self._get_context(conversation_id)
        context.append({"role": role, "content": content})
        
        # Trim context if too long
        if len(context) > 20:  # Adjust based on needs
            context.pop(0)
            
    def _build_message_array(self, context: List[Dict[str, str]], user_id: int, message: str) -> List[Dict[str, str]]:
        # Add current time to system prompt
        current_time = f"\nCurrent time: {datetime.now().strftime('%B %d %Y %I:%M %p')}"
        
        messages = [
            {"role": "system", "content": self.config["system_prompt"] + current_time},
            *context,
            {"role": "user", "content": message}
        ]
        return messages
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current system time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memories",
                    "description": "Search through messages with all users for additional context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories"
                            },
                            "min_similarity": {
                                "type": "number",
                                "description": "Minimum similarity threshold (between 0 and 1)",
                                "default": 0.6
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            
            # Add tool call to results
            results.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args)
                    }
                }]
            })
            
            # Execute tool and add result
            try:
                if tool_name == "get_current_time":
                    result = {
                        "current_time": datetime.now().strftime("%I:%M %p"),
                        "date": datetime.now().strftime("%B %d, %Y")
                    }
                elif tool_name == "search_memories":
                    result = self.memory.search_memories(**args)
                else:
                    result = {"error": "Unknown tool"}
                
                results.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call["id"],
                    "name": tool_name
                })
                
            except Exception as e:
                logging.error(f"Error executing tool {tool_name}: {e}")
                results.append({
                    "role": "tool",
                    "content": json.dumps({"error": str(e)}),
                    "tool_call_id": tool_call["id"],
                    "name": tool_name
                })
                
        return results
