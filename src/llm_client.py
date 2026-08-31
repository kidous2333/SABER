"""
llm_client.py
Wrapper for LLM calls, providing a unified interface.
Supports providers: openai / glm / anthropic.
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client with retry support. Provider can be openai / glm / anthropic."""

    def __init__(self, cfg: dict):
        llm_cfg = cfg["llm"]
        self.provider = llm_cfg.get("provider", "openai").lower()
        self.model = llm_cfg.get("model", "gpt-4o")
        self.temperature = llm_cfg.get("temperature", 0.7)
        self.max_tokens = llm_cfg.get("max_tokens", 4096)

        if self.provider == "anthropic":
            import anthropic
            api_key = llm_cfg.get("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
            base_url = llm_cfg.get("base_url", "https://api.anthropic.com")
            user_agent = llm_cfg.get(
                "user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0",
            )
            headers = {"User-Agent": user_agent}
            custom_headers = llm_cfg.get("custom_headers", {})
            if isinstance(custom_headers, dict):
                headers.update(custom_headers)
            self.client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url,
                default_headers=headers,
            )
            self._stream_timeout = llm_cfg.get("stream_timeout", 300)
            self._thinking_budget = llm_cfg.get("thinking_budget", 0)
            self._call = self._call_anthropic

        elif self.provider == "glm":
            from zhipuai import ZhipuAI
            api_key = llm_cfg.get("api_key") or os.getenv("ZHIPUAI_API_KEY", "")
            self.client = ZhipuAI(api_key=api_key)
            self._call = self._call_openai_compat

        else:  # openai
            from openai import OpenAI
            api_key = llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
            user_agent = llm_cfg.get(
                "user_agent",
                "codex_cli_rs/0.77.0 (Windows 10.0.26100; x86_64) WindowsTerminal",
            )
            headers = {"User-Agent": user_agent}
            # Merge custom headers from config
            custom_headers = llm_cfg.get("custom_headers", {})
            if isinstance(custom_headers, dict):
                headers.update(custom_headers)
            self.client = OpenAI(
                api_key=api_key,
                base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
                timeout=llm_cfg.get("timeout", 120.0),
                default_headers=headers,
            )
            self._call = self._call_openai_compat

    def _call_openai_compat(self, messages, temperature, max_tokens) -> str:
        t0 = time.time()
        logger.info(
            f"[LLM] Sending OpenAI request (model={self.model}, "
            f"max_tokens={max_tokens})..."
        )
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            text_parts = []
            reasoning_parts = []
            chunk_count = 0
            last_log_time = t0
            finish_reason = None
            logged_first_fields = False
            for chunk in stream:
                now = time.time()
                chunk_count += 1
                if chunk_count == 1:
                    logger.info(f"[LLM] Received first chunk, wait time {now - t0:.1f}s")
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                # Print delta fields on first occurrence for diagnostics
                if not logged_first_fields and delta:
                    fields = [k for k, v in (delta.__dict__ if hasattr(delta, '__dict__') else {}).items() if v and not k.startswith('_')]
                    if fields:
                        logger.info(f"[LLM] delta fields: {fields}")
                        logged_first_fields = True
                # Normal text content
                content = getattr(delta, "content", None)
                if content:
                    text_parts.append(content)
                # reasoning_content (some proxies use this field for thinking)
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                # Log progress every 15 seconds
                if (now - last_log_time) >= 15:
                    logger.info(
                        f"[LLM] Stream progress [{now - t0:.0f}s]: chunks={chunk_count}, "
                        f"text={len(''.join(text_parts))} chars, "
                        f"reasoning={len(''.join(reasoning_parts))} chars"
                    )
                    last_log_time = now

            elapsed = time.time() - t0
            text = "".join(text_parts).strip()
            reasoning_text = "".join(reasoning_parts).strip()
            logger.info(
                f"[LLM] OpenAI completed [{elapsed:.1f}s]: text={len(text)} chars, "
                f"reasoning={len(reasoning_text)} chars, "
                f"chunks={chunk_count}, finish_reason={finish_reason}"
            )
            # If text is empty but reasoning exists, attempt to extract JSON from reasoning
            if not text and reasoning_text:
                logger.warning(
                    f"[LLM] text is empty, attempting to extract result from reasoning ({len(reasoning_text)} chars)..."
                )
                logger.info(f"[LLM] First 500 chars of reasoning:\n{reasoning_text[:500]}")
                import re
                # Try to find a JSON array (pattern starting with [ followed by {)
                json_match = re.search(r"\[\s*\{", reasoning_text)
                if json_match:
                    start = json_match.start()
                    # Find the matching ] from this position
                    bracket_depth = 0
                    end = start
                    for i in range(start, len(reasoning_text)):
                        if reasoning_text[i] == '[':
                            bracket_depth += 1
                        elif reasoning_text[i] == ']':
                            bracket_depth -= 1
                            if bracket_depth == 0:
                                end = i + 1
                                break
                    if end > start:
                        text = reasoning_text[start:end]
                        logger.info(f"[LLM] Extracted JSON from reasoning ({len(text)} chars)")
                    else:
                        # Truncated case, take until end
                        text = reasoning_text[start:]
                        logger.info(f"[LLM] Extracted truncated JSON from reasoning ({len(text)} chars)")
                else:
                    logger.warning(
                        f"[LLM] No JSON array found in reasoning, "
                        f"first 200 chars: {reasoning_text[:200]}"
                    )
            elif not text:
                logger.warning("[LLM] OpenAI returned completely empty response")
            return text
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[LLM] OpenAI request error ({elapsed:.1f}s): {type(e).__name__}: {e}")
            raise

    def _extract_text_response(self, response) -> str:
        """Extract text content from different SDK / proxy return values."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict):
            if "choices" in response:
                choices = response.get("choices") or []
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        message = choice.get("message") or {}
                        if isinstance(message, dict):
                            content = message.get("content", "")
                        else:
                            content = getattr(message, "content", "")
                    else:
                        message = getattr(choice, "message", None)
                        content = getattr(message, "content", "") if message is not None else ""
                    return str(content).strip()
            content = response.get("content", "")
            return str(content).strip()

        choices = getattr(response, "choices", None)
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                content = getattr(message, "content", "")
                return str(content).strip()

        content = getattr(response, "content", None)
        if content is not None:
            if isinstance(content, list):
                parts = []
                for item in content:
                    text = getattr(item, "text", None)
                    if text is None and isinstance(item, dict):
                        text = item.get("text", "")
                    if text:
                        parts.append(str(text))
                return "".join(parts).strip()
            return str(content).strip()

        return str(response).strip()

    def _call_anthropic(self, messages, temperature, max_tokens) -> str:
        # Anthropic API requires system message to be passed separately
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        kwargs = dict(
            model=self.model,
            messages=filtered,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        # Extended thinking control
        thinking_budget = getattr(self, "_thinking_budget", 0)
        if thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # When thinking is enabled, temperature must be 1 (Anthropic requirement)
            kwargs["temperature"] = 1
            logger.info(f"[LLM] thinking enabled, budget_tokens={thinking_budget}")
        else:
            kwargs["thinking"] = {"type": "disabled"}
            logger.info("[LLM] thinking disabled")

        stream_timeout = getattr(self, "_stream_timeout", 300)
        text_parts = []
        thinking_parts = []
        thinking_chars = 0
        block_types = []
        stop_reason = None
        t0 = time.time()
        event_count = 0
        text_chunk_count = 0
        last_log_time = t0

        logger.info(
            f"[LLM] Anthropic streaming request started (model={self.model}, "
            f"max_tokens={max_tokens}, timeout={stream_timeout}s)..."
        )

        try:
            with self.client.messages.stream(**kwargs) as stream:
                logger.info("[LLM] Stream connection established, waiting for events...")
                for event in stream:
                    now = time.time()
                    elapsed = now - t0
                    event_count += 1
                    event_type = getattr(event, "type", str(type(event).__name__))

                    # Timeout check
                    if elapsed > stream_timeout:
                        logger.error(
                            f"[LLM] Stream timeout ({elapsed:.1f}s > {stream_timeout}s), "
                            f"events={event_count}, text_chunks={text_chunk_count}, "
                            f"text={len(''.join(text_parts))} chars, thinking={thinking_chars} chars"
                        )
                        break

                    # First event
                    if event_count == 1:
                        logger.info(
                            f"[LLM] Received first event: type={event_type}, "
                            f"wait time {elapsed:.1f}s"
                        )

                    # Handle different event types
                    if event_type == "content_block_start":
                        cb = getattr(event, "content_block", None)
                        btype = getattr(cb, "type", "?") if cb else "?"
                        block_types.append(btype)
                        logger.info(f"[LLM] New block started: type={btype}")

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", "?") if delta else "?"
                        if delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
                                text_parts.append(text)
                                text_chunk_count += 1
                        elif delta_type == "thinking_delta":
                            thinking = getattr(delta, "thinking", "")
                            if thinking:
                                thinking_parts.append(thinking)
                                thinking_chars += len(thinking)

                    elif event_type == "message_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            stop_reason = getattr(delta, "stop_reason", None)

                    # Log progress every 15 seconds
                    if (now - last_log_time) >= 15:
                        logger.info(
                            f"[LLM] Stream progress [{elapsed:.0f}s]: events={event_count}, "
                            f"text={len(''.join(text_parts))} chars, "
                            f"thinking={thinking_chars} chars, blocks={block_types}"
                        )
                        last_log_time = now

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"[LLM] Stream exception ({elapsed:.1f}s): {type(e).__name__}: {e}"
            )
            raise

        elapsed_total = time.time() - t0
        result = "".join(text_parts).strip()
        thinking_text = "".join(thinking_parts).strip()
        logger.info(
            f"[LLM] Stream completed [{elapsed_total:.1f}s]: text={len(result)} chars, "
            f"thinking={thinking_chars} chars, events={event_count}, "
            f"stop_reason={stop_reason}, blocks={block_types}"
        )
        if not result:
            # Print thinking content for diagnostics
            if thinking_text:
                logger.warning(
                    f"[LLM] No text output, but thinking content exists ({len(thinking_text)} chars). "
                    f"Attempting to extract result from thinking..."
                )
                logger.info(f"[LLM] First 500 chars of thinking content:\n{thinking_text[:500]}")
                # Fallback: if thinking contains a JSON array, use it directly as the result
                import re
                json_match = re.search(r"\[.*\]", thinking_text, re.DOTALL)
                if json_match:
                    result = json_match.group()
                    logger.info(
                        f"[LLM] Extracted JSON from thinking ({len(result)} chars), using as fallback"
                    )
                else:
                    logger.warning(
                        f"[LLM] No JSON array found in thinking, content summary:\n"
                        f"  first 200 chars: {thinking_text[:200]}\n"
                        f"  last 200 chars: {thinking_text[-200:]}"
                    )
            else:
                logger.warning(
                    f"[LLM] Anthropic returned completely empty response! stop_reason={stop_reason}, "
                    f"blocks={block_types}, elapsed={elapsed_total:.1f}s"
                )
        return result

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        """Send a chat request and return the model's text reply; auto-retries on failure or empty response."""
        t = temperature if temperature is not None else self.temperature
        m = max_tokens if max_tokens is not None else self.max_tokens
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                result = self._call(messages, t, m)
                if result:
                    return result
                logger.warning(
                    f"LLM returned empty content (attempt {attempt + 1}), retrying..."
                )
            except Exception as e:
                last_err = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
        if last_err is not None:
            raise last_err
        return ""
