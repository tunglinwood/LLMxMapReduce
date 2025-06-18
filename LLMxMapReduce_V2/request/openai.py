import os
from openai import OpenAI, InternalServerError, RateLimitError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
    retry_if_exception_type
)
import logging
logger = logging.getLogger(__name__)


class OpenAIRequest:
    def __init__(self, model):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
        )
        self.model = model

    def _validate_messages_length(self, messages):
        """Validate and truncate messages to fit within model's context length"""
        max_length = 140000 * 4  # Approximate 4 chars per token
        total_length = 0
        valid_messages = []
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else msg['content']
            if total_length + len(content) > max_length:
                remaining = max_length - total_length
                if remaining > 1000:  # Only truncate if we can keep meaningful content
                    if hasattr(msg, 'content'):
                        msg.content = content[:remaining] + '... [truncated]'
                    else:
                        msg['content'] = content[:remaining] + '... [truncated]'
                    valid_messages.append(msg)
                break
            valid_messages.append(msg)
            total_length += len(content)
            
        logger.debug(f"Validated messages length: {total_length} chars")
        return valid_messages

    @retry(
        wait=wait_random_exponential(multiplier=2, max=60),
        stop=stop_after_attempt(100),
        retry=retry_if_exception_type((RateLimitError, InternalServerError, APIError))
        )
    def completion(self, messages, **kwargs):
        try:
            messages = self._validate_messages_length(messages)
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )
            # 新增检查：确保响应包含有效的 choices 数据
            if not response.choices or len(response.choices) == 0:
                error_msg = "OpenAI API returned empty choices in response"
                logger.debug(error_msg)
                raise ValueError(error_msg)
            answer = response.choices[0].message.content
            token_usage = response.usage

        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded in OpenAIRequest.completion: {e}")
            raise 
        except InternalServerError as e:
            logger.warning(f"Internal server error in OpenAIRequest.completion: {e}")
            # logger.warning(f"Prompt: {messages}")
            raise 
        except Exception as e:
            logger.error(f"Unexpected error in OpenAIRequest.completion: {e}. messages: \n{messages}")
            raise 
                
        return answer, token_usage
