"""Anthropic Messages API ↔ AWS Bedrock proxy for the yuji verifier judge.

The bundled rubric_judge_cli.py (shipped in the ECR judge image) only speaks
JUDGE_WIRE = openai | anthropic | gemini. For CC_MODE=bedrock, this proxy
listens on 127.0.0.1 as an Anthropic Messages API server and forwards each
request to AWS Bedrock via litellm. The judge thinks it is calling
api.anthropic.com; the actual LLM is the Bedrock inference-profile ARN that
JUDGE_MODEL resolves to through BEDROCK_MODEL_IDS.

Zero AWS credentials ever leave the host: the proxy process holds them and
the verifier container only sees ANTHROPIC_BASE_URL pointing at the proxy.
"""

from .server import BedrockJudgeServer, resolve_bedrock_arn

__all__ = ["BedrockJudgeServer", "resolve_bedrock_arn"]
