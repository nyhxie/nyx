from datetime import datetime
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from Memory.memory_handler import MemoryHandler
import pytz

class AIHandler:
    def __init__(self, config: dict, memory_handler: MemoryHandler):
        self.config = config
        self.memory = memory_handler
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Cache for conversation contexts
        self.conversation_contexts = {}
        
    async def get_response(self, user_id: int, message: str, conversation_id: str) -> str:
        if conversation_id not in self.conversation_contexts:
            conversation_history = self.memory.rebuild_conversation_chain(user_id)
            self.conversation_contexts[conversation_id] = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in conversation_history
            ]
        
        context = self._get_context(conversation_id)
        messages = self._build_message_array(context, user_id, message)
        api_url = f"{self.config['providers']['lmstudio']['base_url']}/chat/completions"

        try:
            # Initial request with tools enabled
            response = await self.client.post(api_url, json={
                "messages": messages,
                "model": self.config["model"],
                "tools": self._get_available_tools(),
                "tool_choice": "auto",
                **self.config["extra_api_parameters"]
            })
            response.raise_for_status()
            initial_result = response.json()
            assistant_message = initial_result["choices"][0]["message"]

            # Check for tool calls
            if tool_calls := assistant_message.get("tool_calls"):
                # Store the tool call request message
                tool_call_message = {
                    "role": "assistant",
                    "content": None,  # Set content to None since we're using tool_calls
                    "tool_calls": [{
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": tool_call["function"]
                    } for tool_call in tool_calls]
                }
                messages.append(tool_call_message)

                # Execute tools and add results
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])

                    # Execute tool
                    if tool_name == "search_memories":
                        result = self.memory.search_memories(**args)
                    elif tool_name == "get_current_time":
                        result = {
                            "current_time": datetime.now().strftime("%I:%M %p"),
                            "date": datetime.now().strftime("%B %d, %Y")
                        }

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": tool_call["id"],
                        "name": tool_name
                    })

                # Make final request with tool results AND tools enabled
                final_response = await self.client.post(api_url, json={
                    "messages": messages,
                    "model": self.config["model"],
                    "tools": self._get_available_tools(),  # Keep tools available
                    "tool_choice": "auto",
                    **self.config["extra_api_parameters"]
                })
                final_result = final_response.json()
                final_message = final_result["choices"][0]["message"]
                
                # If we got another tool call, process it recursively
                if final_message.get("tool_calls"):
                    messages.append(final_message)
                    return await self.get_response(user_id, message, conversation_id)
                
                return final_message["content"]
            else:
                content = assistant_message["content"]

            # Update context with final result
            self._update_context(conversation_id, message, "user")
            self._update_context(conversation_id, content, "assistant")
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
        
        # Keep a reasonable context window
        if len(context) > 100:  # Increased from 20 to maintain more context
            context.pop(0)
            
    def _build_message_array(self, context: List[Dict[str, str]], user_id: int, message: str) -> List[Dict[str, str]]:
        # Add current time to system prompt using EST
        est = pytz.timezone('America/New_York')
        current_time = f"\nCurrent time: {datetime.now(est).strftime('%B %d %Y %I:%M %p')}"
        
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
                    result = self._get_current_time()
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

    def _get_current_time(self) -> Dict[str, str]:
        # Get EST timezone
        est = pytz.timezone('America/New_York')
        current_time = datetime.now(est)
        return {
            "current_time": current_time.strftime("%I:%M %p"),
            "date": current_time.strftime("%B %d, %Y")
        }
