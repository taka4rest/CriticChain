import os
from dotenv import load_dotenv

# Load env vars before importing modules that depend on them (like prompts.py)
load_dotenv()

from typing import TypedDict, List, Optional, Annotated, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import operator
from logger_config import setup_logger, log_token_usage
from browser_search import BrowserSearchRun
from database import get_db, ReviewSession
import json
import subprocess

# Import specialized prompts
from prompts import (
    ROUTER_SYSTEM_PROMPT,
    LINT_PROMPT_ENGINEERING_SYSTEM,
    LINT_PROGRAMMING_SYSTEM,
    ARCHITECTURE_PROMPT_ENGINEERING_SYSTEM,
    ARCHITECTURE_PROGRAMMING_SYSTEM,
    DRAFT_REVIEW_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    MANAGER_SYSTEM_PROMPT,
    MANAGER_CONSENSUS_PROMPT,
    SCORING_SYSTEM_PROMPT
)

# Import JSON schemas for with_structured_output
from schemas import (
    LintOutput,
    AnalyzeOutput,
    FactCheckOutput,
    CritiqueOutput,
    ResearchOutput
)

# --- Logger Setup ---
logger = setup_logger("graph_agent")

def overwrite(left, right):
    return right

def escape_for_prompt(s: str) -> str:
    """Escape { and } for LangChain prompt templates."""
    if not s:
        return s
    return s.replace('{', '{{').replace('}', '}}')

# --- Constants ---
MAX_REFINE_ITERATIONS = int(os.getenv("MAX_REFINE_ITERATIONS", "5"))  # Maximum critique → refine loop iterations

# --- Pricing & Input Validation ---
_BILLING_ENABLED = os.getenv("CRITIC_TIER", "auto") != "lite"

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "100000"))  # System limit
TOKENS_PER_CHAR = float(os.getenv("TOKENS_PER_CHAR", "0.4"))  # Avg for mixed JP/EN

import json

if _BILLING_ENABLED:
    CREDIT_PRICE_USD = float(os.getenv("CREDIT_PRICE_USD", "3.0"))
    DEFAULT_TIERS = json.dumps([
        {"max_tokens": 5000, "credits": 1},
        {"max_tokens": 20000, "credits": 2},
        {"max_tokens": 50000, "credits": 4},
        {"max_tokens": 100000, "credits": 8}
    ])
    PRICING_TIERS = json.loads(os.getenv("PRICING_TIERS", DEFAULT_TIERS))
else:
    CREDIT_PRICE_USD = 0
    PRICING_TIERS = []


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. Japanese ≈ 0.5-1 token/char, English ≈ 0.25 token/char."""
    return int(len(text) * TOKENS_PER_CHAR)


def calculate_credits(tokens: int) -> int:
    """Calculate credit consumption based on token count. Returns -1 if over limit."""
    for tier in PRICING_TIERS:
        if tokens <= tier["max_tokens"]:
            return tier["credits"]
    return -1  # Over limit


def get_tier_name(credits: int) -> str:
    """Get human-readable tier name."""
    tier_names = {1: "Small", 2: "Medium", 4: "Large", 8: "Enterprise"}
    return tier_names.get(credits, "Custom")


def validate_input(submission: str, original_code: str = "", model_answer: str = "") -> dict:
    """
    Validate input and calculate pricing (Enterprise) or just size (Lite/OSS).

    Returns:
        dict with keys: valid, tokens, credits, tier, cost_usd, message
    """
    total_text = (submission or "") + (original_code or "") + (model_answer or "")
    tokens = estimate_tokens(total_text)

    if not _BILLING_ENABLED:
        if tokens > MAX_INPUT_TOKENS:
            return {
                "valid": False, "tokens": tokens, "credits": 0,
                "tier": "Rejected", "cost_usd": 0,
                "message": f"⚠️ Input too large ({tokens:,} tokens, limit: {MAX_INPUT_TOKENS:,})"
            }
        return {
            "valid": True, "tokens": tokens, "credits": 0,
            "tier": "OSS", "cost_usd": 0,
            "message": f"✅ {tokens:,} tokens"
        }

    credits = calculate_credits(tokens)
    if credits == -1:
        return {
            "valid": False, "tokens": tokens, "credits": 0,
            "tier": "Rejected", "cost_usd": 0,
            "message": f"⚠️ Input too large ({tokens:,} tokens, limit: {MAX_INPUT_TOKENS:,})"
        }

    cost_usd = credits * CREDIT_PRICE_USD
    tier_name = get_tier_name(credits)
    return {
        "valid": True, "tokens": tokens, "credits": credits,
        "tier": tier_name, "cost_usd": cost_usd,
        "message": f"✅ {tier_name}: {credits} credits (${cost_usd:.2f})"
    }

# --- State Definition ---
class ReviewState(TypedDict):
    submission: str
    original_code: Optional[str] # Added for Diff tasks
    task_type: str # "Prompt Engineering" or "Programming"
    model_answer: Optional[str]
    analysis_result: Optional[Any] # Changed to Any to store dict with metadata
    fact_check_result: Optional[Any]
    draft_review: Optional[Any]
    critique_comment: Optional[Any] # For Multi-Agent Debate
    evaluation_result: Optional[Any] # For LLM-as-a-Judge
    lint_result: Optional[Any] # For Linter
    research_result: Optional[Any] # For Researcher
    consistency_result: Optional[Any] # For Consistency Check
    messages: Annotated[List[BaseMessage], operator.add]
    human_feedback: Optional[str]
    status: Annotated[str, overwrite]
    use_cli_bypass: Optional[bool] # Feature Flag (Legacy)
    review_mode: Optional[str] # "standard", "hybrid", "consensus"
    cli_output: Optional[str] # Output from CLI Bypass Node
    disable_web_search: Optional[bool] # Feature Flag to disable web search
    refine_count: Optional[int] # Counter for critique → refine loop iterations
    # --- Billing Fields ---
    credits_charged: Optional[int] # Credits consumed for this review
    pricing_info: Optional[dict] # Full pricing details (tokens, tier, cost_usd)
    # --- Scoring Fields ---
    score: Optional[int] # Final score (0-100)
# --- LLM Setup (Multi-Model) ---
def get_llm(model_env_var: str):
    # Use newer flash model as default to avoid 404 on old pro models
    model_name = os.getenv(model_env_var, "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

llm_router = get_llm("MODEL_ROUTER") # New model for routing
llm_lint = get_llm("MODEL_LINT")
llm_analyze = get_llm("MODEL_ANALYZE")
llm_fact_check = get_llm("MODEL_FACT_CHECK")
llm_draft = get_llm("MODEL_DRAFT")
llm_critique = get_llm("MODEL_CRITIQUE")
llm_refine = get_llm("MODEL_REFINE")
llm_consistency = get_llm("MODEL_CONSISTENCY")
llm_judge = get_llm("MODEL_JUDGE")
llm_research = get_llm("MODEL_RESEARCH")

# --- Tools ---
# Using Playwright-based browser search (local, free, stable)
try:
    search = BrowserSearchRun()
except Exception as e:
    print(f"Warning: Browser search init failed: {e}")
    search = None

# --- Helper for Robust LLM Invocation ---
def invoke_llm_with_fallback(chain, inputs, node_name, model_env_var, timeout=90, max_retries=3):
    """
    Invoke LLM with timeout, retry, and fallback to alternative model.

    Args:
        chain: LangChain chain to invoke
        inputs: Input dictionary for the chain
        node_name: Name of the node (for logging)
        model_env_var: Environment variable name for the primary model
        timeout: Timeout in seconds (default 90)
        max_retries: Maximum number of retries (default 3)

    Returns:
        LLM result object

    Raises:
        Exception: If all retries and fallback fail
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    primary_model = os.getenv(model_env_var, "gemini-2.5-flash")
    fallback_model = os.getenv("FALLBACK_MODEL", "gemini-2.5-pro")

    # --- Input Data Logging for Diagnostics ---
    logger.info(f"{node_name}: ===== INPUT DATA ANALYSIS =====")
    total_chars = 0
    for key, value in inputs.items():
        value_str = str(value)
        char_count = len(value_str)
        estimated_tokens = char_count // 4  # Rough estimate: 4 chars = 1 token
        total_chars += char_count

        # Preview first 200 chars
        preview = value_str[:200] + "..." if len(value_str) > 200 else value_str

        logger.info(f"{node_name}: Input[{key}] - {char_count} chars (~{estimated_tokens} tokens)")
        logger.debug(f"{node_name}: Input[{key}] Preview: {preview}")

    total_estimated_tokens = total_chars // 4
    logger.info(f"{node_name}: TOTAL INPUT - {total_chars} chars (~{total_estimated_tokens} tokens)")

    # Warn if input is very large
    if total_estimated_tokens > 30000:
        logger.warning(f"{node_name}: ⚠️ Large input detected ({total_estimated_tokens} tokens). This may cause timeout.")
    elif total_estimated_tokens > 50000:
        logger.error(f"{node_name}: ⚠️⚠️ Very large input ({total_estimated_tokens} tokens)! High risk of timeout.")

    logger.info(f"{node_name}: ==============================")
    # --- End Input Logging ---

    def _invoke():
        return chain.invoke(inputs)

    # Try primary model with retries
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"{node_name}: Attempt {attempt}/{max_retries} with {primary_model} (timeout: {timeout}s)")

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke)
                try:
                    result = future.result(timeout=timeout)
                    logger.info(f"{node_name}: Success on attempt {attempt}")
                    return result
                except FutureTimeoutError:
                    logger.warning(f"{node_name}: Timeout ({timeout}s) on attempt {attempt}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"{node_name}: Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    continue

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"{node_name}: Error on attempt {attempt}: {error_msg}")

            # Check if it's a DeadlineExceeded or timeout-related error
            if "DeadlineExceeded" in error_msg or "504" in error_msg or "Timeout" in error_msg:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"{node_name}: Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            # For other errors, re-raise immediately
            else:
                raise

    # All retries failed, try fallback model
    logger.warning(f"{node_name}: All retries with {primary_model} failed. Trying fallback model: {fallback_model}")

    try:
        # Reconstruct chain with fallback model
        from langchain_google_genai import ChatGoogleGenerativeAI
        fallback_llm = ChatGoogleGenerativeAI(model=fallback_model, temperature=0)

        # Extract the prompt template from the chain
        # Assuming chain is a simple prompt | llm chain
        if hasattr(chain, 'first') and hasattr(chain, 'last'):
            prompt_template = chain.first
            fallback_chain = prompt_template | fallback_llm
        else:
            raise ValueError("Cannot extract prompt from chain for fallback")

        logger.info(f"{node_name}: Invoking fallback model {fallback_model}")
        result = fallback_chain.invoke(inputs)
        logger.info(f"{node_name}: Fallback model succeeded")
        return result

    except Exception as fallback_error:
        logger.error(f"{node_name}: Fallback model also failed: {fallback_error}")
        raise Exception(f"Both primary ({primary_model}) and fallback ({fallback_model}) models failed for {node_name}") from fallback_error

# --- Helper for Token Logging ---
def record_usage(result, node_name, model_env_var, config=None, session_id=None):
    """Helper to extract and log token usage. Returns usage dict.

    Args:
        result: LLM result with usage_metadata
        node_name: Name of the node
        model_env_var: Environment variable for model name
        config: RunnableConfig (optional, used to extract thread_id if session_id is None)
        session_id: Database session ID (ReviewSession.id) for per-review tracking
    """
    usage_dict = {"input": 0, "output": 0, "total": 0}
    try:
        # Fallback to thread_id if session_id is missing
        if session_id is None and config:
            session_id = config["configurable"].get("thread_id")

        usage = result.usage_metadata
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            usage_dict = {"input": input_tokens, "output": output_tokens, "total": total_tokens}

            model_name = os.getenv(model_env_var, "unknown-model")


            logger.info(f"{node_name} Completed. Tokens: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")
            log_token_usage(node_name, model_name, input_tokens, output_tokens, session_id=session_id)
            logger.info(f"{node_name}: Successfully called log_token_usage")
    except Exception as e:
        logger.warning(f"Failed to log token usage for {node_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return usage_dict

def clean_output_for_logging(content):
    """
    Clean LLM output by removing Base64 signatures from extras field.
    This makes DEBUG logs more readable.

    Args:
        content: LLM result content (can be string, list, or dict)

    Returns:
        Cleaned content for logging
    """
    if isinstance(content, list):
        cleaned_list = []
        for item in content:
            if isinstance(item, dict):
                # Remove 'extras' field which contains Base64 signature
                cleaned_item = {k: v for k, v in item.items() if k != 'extras'}
                cleaned_list.append(cleaned_item)
            else:
                cleaned_list.append(item)
        return str(cleaned_list)
    elif isinstance(content, dict):
        # Remove 'extras' field
        cleaned_dict = {k: v for k, v in content.items() if k != 'extras'}
        return str(cleaned_dict)
    else:
        return str(content)

# --- Node Functions ---

def router_node(state: ReviewState, config: RunnableConfig):
    """Classifies the submission into a task type AND validates/charges billing."""
    logger.info("--- Router Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    submission = state["submission"]
    original_code = state.get("original_code", "")
    model_answer = state.get("model_answer", "")

    # Log input data for audit trail (Debug level, full content)
    logger.debug(f"=== INPUT DATA START ===")
    logger.debug(f"Submission ({len(submission)} chars):")
    logger.debug(submission)
    logger.debug(f"=== INPUT DATA END ===")

    # === SERVER-SIDE BILLING (Single Source of Truth) ===
    pricing = validate_input(submission, original_code=original_code, model_answer=model_answer)

    if not pricing["valid"]:
        logger.warning(f"❌ Input REJECTED: {pricing['message']}")
        return {
            "status": "rejected",
            "pricing_info": pricing,
            "credits_charged": 0,
            "messages": [AIMessage(content=f"❌ Review Rejected: {pricing['message']}")]
        }

    # Record input / billing info
    if _BILLING_ENABLED:
        logger.info(f"💰 BILLING: {pricing['tier']} tier - {pricing['credits']} credits (${pricing['cost_usd']:.2f})")
    logger.info(f"   Input: {pricing['tokens']:,} tokens ({len(submission):,} chars)")

    # If task_type is already manually set (e.g. via UI), respect it.
    if state.get("task_type") and state["task_type"] != "Auto":
        logger.info(f"Task type already set to: {state['task_type']}")
        return {
            "task_type": state["task_type"],
            "pricing_info": pricing,
            "credits_charged": pricing["credits"]
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("user", "提出物:\n{submission}")
    ])

    chain = prompt | llm_router
    result = chain.invoke({"submission": submission})
    usage = record_usage(result, "Router", "MODEL_ROUTER", config=config, session_id=session_id)

    task_type = "Prompt Engineering" # Default
    try:
        content = result.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        data = json.loads(content.strip())
        task_type = data.get("task_type", "Prompt Engineering")
    except Exception as e:
        logger.error(f"Router parsing failed: {e}. Defaulting to Prompt Engineering.")

    logger.info(f"Router determined task type: {task_type}")

    return {
        "task_type": task_type,
        "pricing_info": pricing,
        "credits_charged": pricing["credits"],
        "messages": [AIMessage(content=f"Router: Detected task type as {task_type}")]
    }

def get_text_content(content: Any) -> str:
    """Helper to extract pure text from LLM content, stripping signatures."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join([str(part.get("text", part) if isinstance(part, dict) else part) for part in content])
    return str(content)

def cli_bypass_node(state: ReviewState, config: RunnableConfig):
    """Executes the review using the local gemini CLI tool via pty."""
    logger.info("--- CLI Bypass Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    submission = state["submission"]
    model_answer = state.get("model_answer", "N/A")
    original_code = state.get("original_code", "N/A")

    review_content = ""

    # Maximum allowed output size (characters)
    MAX_OUTPUT_SIZE = 100000  # 100KB (approx 25k tokens)

    try:
        # Load System Prompt from file (Enterprise → Lite fallback, like prompts.py)
        enterprise_path = "assets/cli-review-prompt_v3.1.md"
        lite_path = "assets/cli-review-prompt_lite.md"

        try:
            with open(enterprise_path, "r") as f:
                system_prompt = f.read()
                logger.info("🔐 CLI: Using Enterprise review prompt")
        except FileNotFoundError:
            try:
                with open(lite_path, "r") as f:
                    system_prompt = f.read()
                    logger.info("📦 CLI: Using Lite review prompt (OSS version)")
            except FileNotFoundError:
                logger.warning(f"No review prompt files found. Using minimal default.")
                system_prompt = "あなたはコードレビューの専門家です。"

        # Construct Combined Prompt with XML tags to prevent prompt injection
        # The template expects data wrapped in XML tags: <original_code>, <submission>, <model_answer>
        # This prevents student prompts from being executed as instructions
        combined_prompt = system_prompt.replace(
            "<original_code>\n```\n\n```\n</original_code>",
            f"""<original_code>
{original_code}
</original_code>"""
        ).replace(
            "<submission>\n```\n\n```\n</submission>",
            f"""<submission>
{submission}
</submission>"""
        ).replace(
            "<model_answer>\n```\n\n```\n</model_answer>",
            f"""<model_answer>
{model_answer}
</model_answer>"""
        ).replace(
            "<review_draft>\n```\n\n```\n</review_draft>",

            """```

```"""  # Empty for now
        )

        # Log prompt details
        logger.info(f"CLI Input - Prompt length: {len(combined_prompt)} chars")
        logger.debug(f"CLI Input - Full content: {combined_prompt}")

        # Prepare logging directory and filename
        from datetime import datetime
        import uuid

        log_dir = "logs/cli-log"
        os.makedirs(log_dir, exist_ok=True)

        # Generate filename: YYYYMMDD_HHMMSS_sessionid.txt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = state.get("thread_id", str(uuid.uuid4())[:8])
        log_filename = f"{timestamp}_{session_id}.txt"
        log_filepath = os.path.join(log_dir, log_filename)

        # Execute via gemini_cli_wrapper module
        from experiments.cli.gemini_cli_wrapper import execute_gemini_cli

        logger.info("Executing gemini CLI via pty wrapper...")
        result = execute_gemini_cli(
            prompt=combined_prompt,
            timeout=180,  # 3 minutes
            model=os.getenv("MODEL_CLI", None)  # Use env var if set
        )

        if result['success']:
            raw_output = result['output']
            output_size = len(raw_output)

            logger.info(f"CLI execution succeeded. Raw output length: {output_size} chars (~{output_size / 1024:.1f} KB)")
            logger.info(f"CLI Output - Full content:\n{raw_output}")

            # Save full I/O to log file
            try:
                with open(log_filepath, "w", encoding="utf-8") as log_file:
                    log_file.write("=" * 80 + "\n")
                    log_file.write(f"CLI Log - {timestamp}\n")
                    log_file.write(f"Session ID: {session_id}\n")
                    log_file.write("=" * 80 + "\n\n")

                    log_file.write("## INPUT PROMPT\n")
                    log_file.write("-" * 80 + "\n")
                    log_file.write(f"Length: {len(combined_prompt)} chars\n\n")
                    log_file.write(combined_prompt)
                    log_file.write("\n\n" + "=" * 80 + "\n\n")

                    log_file.write("## RAW OUTPUT\n")
                    log_file.write("-" * 80 + "\n")
                    log_file.write(f"Length: {output_size} chars ({output_size / 1024 / 1024:.2f} MB)\n\n")
                    log_file.write(raw_output)
                    log_file.write("\n\n" + "=" * 80 + "\n")

                # Log absolute path for clickability
                abs_log_path = os.path.abspath(log_filepath)
                logger.info(f"CLI I/O saved to: {abs_log_path}")
            except Exception as log_err:
                logger.error(f"Failed to save CLI log: {log_err}")

            # Check if output is abnormally large
            if output_size > MAX_OUTPUT_SIZE:
                logger.warning(f"CLI output too large ({output_size} chars). Trimming to {MAX_OUTPUT_SIZE} chars.")
                logger.warning(f"Original output size: {output_size / 1024 / 1024:.2f} MB")
                logger.warning(f"Full output saved to: {log_filepath}")

                # Take first portion (preserve important content at the beginning)
                review_content = raw_output[:MAX_OUTPUT_SIZE]
                review_content += "\n\n[... Output truncated due to size limit ...]"
                review_content += f"\n[Full output saved to: {log_filepath}]"

                logger.info(f"CLI Output trimmed to: {len(review_content)} chars")
            else:
                review_content = raw_output
        else:
            review_content = f"CLI Error:\n{result['error']}"
            logger.error(f"CLI execution failed: {result['error']}")

    except Exception as e:
        review_content = f"CLI Bypass Failed: {e}"
        logger.error(f"CLI Bypass Exception: {e}")

    # Final size check
    final_size = len(review_content)
    logger.info(f"CLI Output (final): {final_size} chars (~{final_size // 4} tokens estimate)")

    return {
        "cli_output": review_content,  # Store processed CLI output
        "status": "cli_done",
        "messages": [AIMessage(content=f"CLI Review:\n{get_text_content(review_content)}")]
    }

def manager_node(state: ReviewState, config: RunnableConfig):
    """Merges CLI output with Lint and Research results."""
    logger.info("--- Manager Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    cli_output = state.get("cli_output", "")
    lint_data = state.get("lint_result", {})
    lint_result = lint_data.get("content", "") if isinstance(lint_data, dict) else str(lint_data)
    research_data = state.get("research_result", {})
    research_result = research_data.get("content", "") if isinstance(research_data, dict) else str(research_data)

    # Extract Analysis Result (for Consensus Mode)
    analysis_data = state.get("analysis_result", {})
    analysis_result = analysis_data.get("content", "") if isinstance(analysis_data, dict) else str(analysis_data)

    review_mode = state.get("review_mode", "hybrid")

    # review_mode: CLIとAnalyze分析をどう統合するか
    # - "consensus" (or "hybrid"): CLI + Analyze 両方を統合（教育的観点 + 技術的深さ）
    # - "cli_only": CLIのみを使用（Analyzeは無視）
    if review_mode == "consensus" or review_mode == "hybrid":
        logger.info("Using Hybrid Mode: Integrating both CLI and Analyze results")
        system_prompt = MANAGER_CONSENSUS_PROMPT  # = MANAGER_HYBRID_PROMPT
        user_prompt = "CLI分析結果:\n{cli_output}\n\nAnalyze分析結果:\n{analysis_result}\n\nLint結果:\n{lint_result}\n\nリサーチ結果:\n{research_result}"
        input_vars = {
            "cli_output": cli_output,
            "analysis_result": analysis_result,
            "lint_result": lint_result,
            "research_result": research_result
        }
    else:  # cli_only
        logger.info("Using CLI-Only Mode: CLI as primary source, supplemented by Lint & Research")
        system_prompt = MANAGER_SYSTEM_PROMPT  # = MANAGER_CLI_ONLY_PROMPT
        user_prompt = "CLI分析結果:\n{cli_output}\n\nLint結果:\n{lint_result}\n\nリサーチ結果:\n{research_result}"
        input_vars = {
            "cli_output": cli_output,
            "lint_result": lint_result,
            "research_result": research_result
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])

    # Use a smart model for integration (e.g., gemini-1.5-pro)
    chain = prompt | llm_analyze
    result = chain.invoke(input_vars)

    usage = record_usage(result, "Manager", "MODEL_ANALYZE", config=config, session_id=session_id)

    return {
        "draft_review": {"content": get_text_content(result.content), "token_usage": usage},
        "analysis_result": {"content": cli_output, "token_usage": {"total": 0}}, # Use CLI output as analysis reference
        "status": "drafted", # Ready for Critique
        "messages": [AIMessage(content=f"Manager Integrated Review:\n{get_text_content(result.content)}")]
    }

def lint_check_node(state: ReviewState, config: RunnableConfig):
    """Checks for syntax errors and variable definitions. Outputs structured JSON via with_structured_output."""
    logger.info("--- Lint Check Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    submission = state["submission"]
    task_type = state["task_type"]
    logger.debug(f"Input - Task Type: {task_type}")

    # Switch prompt based on task type
    if task_type == "Programming":
        system_prompt = LINT_PROGRAMMING_SYSTEM
    else:
        system_prompt = LINT_PROMPT_ENGINEERING_SYSTEM

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "タスクタイプ: {task_type}\n提出物:\n{submission}")
    ])

    # Use with_structured_output for reliable JSON schema enforcement (include_raw=True to get usage metadata)
    structured_llm = llm_lint.with_structured_output(LintOutput, include_raw=True)
    chain = prompt | structured_llm

    try:
        result = chain.invoke({"submission": submission, "task_type": task_type})
        lint_result: LintOutput = result["parsed"]
        raw_message = result["raw"]
        lint_json = lint_result.model_dump()
        raw_content = json.dumps(lint_json, ensure_ascii=False, indent=2)
        # Record token usage from raw message
        usage = record_usage(raw_message, "Lint Check", "MODEL_LINT", config=config, session_id=session_id)
    except Exception as e:
        logger.warning(f"Lint Check: with_structured_output failed, falling back to manual parse: {e}")
        # Fallback to manual parsing
        chain_fallback = prompt | llm_lint
        result = chain_fallback.invoke({"submission": submission, "task_type": task_type})
        # Handle different result types
        if isinstance(result, BaseMessage):
            content_to_parse = result.content
        elif isinstance(result, list):
            content_to_parse = result[0].content if len(result) > 0 and isinstance(result[0], BaseMessage) else str(result)
        else:
            content_to_parse = result
        lint_json = parse_json_output(content_to_parse, "Lint Check")
        raw_content = content_to_parse if isinstance(content_to_parse, str) else str(content_to_parse)
        # Record usage from fallback
        usage = record_usage(result, "Lint Check", "MODEL_LINT", config=config, session_id=session_id)
    logger.info(f"Lint Check: status={lint_json.get('status')}, issues={len(lint_json.get('issues', []))}")
    logger.debug(f"Output - Lint Result (JSON): {json.dumps(lint_json, ensure_ascii=False, indent=2)[:500]}...")

    return {
        "lint_result": {
            "content": raw_content,
            "json": lint_json,
            "token_usage": usage
        },
        "messages": [AIMessage(content=f"Lint Result:\n{get_text_content(raw_content)}")]
    }

def research_node(state: ReviewState, config: RunnableConfig):
    """Searches for best practices based on detected issues. Outputs JSON."""
    logger.info("--- Research Node Entered ---")
    session_id = config["configurable"].get("db_session_id")

    # Check if search is disabled FIRST to save tokens
    disable_web_search = state.get("disable_web_search", False)
    if disable_web_search:
        logger.info("Web search is disabled by user. Skipping search and query generation.")
        research_json = {
            "queries": [],
            "results": [],
            "summary": "Web検索は無効化されています"
        }
        return {
            "research_result": {
                "content": "Web検索は無効化されています。検索結果は利用できません。",
                "json": research_json,
                "token_usage": {"total": 0}
            },
            "messages": [AIMessage(content="Research skipped (disabled).")]
        }

    # 1. Generate or Use Existing Queries
    submission = state["submission"]
    task_type = state["task_type"]
    lint_data = state.get("lint_result", {})
    lint_result = lint_data.get("content", "") if isinstance(lint_data, dict) else str(lint_data)

    # Check if Analyze node already provided queries
    analysis_data = state.get("analysis_result", {})
    analysis_json = analysis_data.get("json", {}) if isinstance(analysis_data, dict) else {}
    research_queries = analysis_json.get("research_queries", [])

    if not research_queries:
        logger.info("Research: No queries from Analyze. Generating default queries...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは技術リサーチャーです。提出物を分析し、**具体的な問題点やトピックに関連する**検索クエリを生成してください。

以下の観点で分析し、検索クエリを生成してください：
1. **検出された問題**: Lint結果から特定された具体的な問題
2. **提出物の特徴**: 提出物が扱っているテーマ（プロンプトチェーン、Pythonの特定ライブラリなど）
3. **セキュリティリスク**: 潜在的リスク

検索クエリは**英語**で、**技術的に具体的**なものにしてください。
検索クエリ文字列のみを3-5個、1行ずつ返してください。"""),
            ("user", """タスクタイプ: {task_type}

Lint結果:
{lint_result}

提出物:
{submission}

上記を踏まえて、具体的な検索クエリを生成してください。""")
        ])

        chain = prompt | llm_research
        query_result = chain.invoke({
            "submission": submission,
            "task_type": task_type,
            "lint_result": lint_result
        })
        query_text = query_result.content
        research_queries = [q.strip() for q in query_text.split('\n') if q.strip()]
        usage = record_usage(query_result, "Research (Query Gen)", "MODEL_RESEARCH", config=config, session_id=session_id)
    else:
        logger.info(f"Research: Using {len(research_queries)} queries from Analyze")
        query_text = "\n".join(research_queries)
        usage = {"input": 0, "output": 0, "total": 0} # No LLM call for query gen

    logger.debug(f"Search Queries: {query_text}")

    # 2. Perform Search
    search_result = ""
    # Double check disable_web_search flag within node
    disable_web_search = state.get("disable_web_search", False)
    if disable_web_search:
        search_result = "Search disabled by user settings."
        logger.info("Research: Search is disabled.")
    elif search and research_queries:
        try:
            combined_results = []
            for q in research_queries[:3]: # Limit to 3 queries
                logger.info(f"Research: Executing search query: {q}")
                res = search.run(q)
                combined_results.append(f"Result for '{q}':\n{res}")
            search_result = "\n\n---\n\n".join(combined_results)
        except Exception as e:
            search_result = f"Search failed: {e}"
            logger.error(f"Search failed: {e}")
    else:
        search_result = "Search tool unavailable or no queries."

    # Build JSON output
    research_json = {
        "queries": [{"query": q, "intent": "dynamic-search"} for q in research_queries],
        "results": [{"source": "web_search", "relevant_content": search_result, "relevance": "high"}],
        "summary": f"Executed {len(research_queries)} queries"
    }

    logger.info(f"Research: queries={len(research_queries)}")
    logger.debug(f"Output - Research Results (JSON): {json.dumps(research_json, ensure_ascii=False)[:500]}...")

    return {
        "research_result": {
            "content": f"Queries:\n{query_text}\n\nResults:\n{search_result}",
            "json": research_json,
            "token_usage": usage,
            "done": True # Mark as done to prevent infinite loop
        },
        "messages": [AIMessage(content=f"Research Results:\n{get_text_content(search_result)}")]
    }

def parse_json_output(content, node_name: str) -> dict:
    """Parse JSON output from LLM, handling markdown code blocks.

    Args:
        content: Can be str, list, or AIMessage object
        node_name: Name of the node for logging
    """
    import json
    import re
    from langchain_core.messages import BaseMessage

    # Handle different input types
    from langchain_core.messages import BaseMessage
    if isinstance(content, list):
        # If it's a list, take the first element (usually AIMessage)
        if len(content) > 0:
            if isinstance(content[0], BaseMessage):
                content = content[0].content
            else:
                content = str(content[0])
        else:
            logger.error(f"{node_name}: Empty list provided to parse_json_output")
            content = ""
    elif isinstance(content, BaseMessage):
        content = content.content
    elif not isinstance(content, str):
        # Convert to string if it's not already
        content = str(content)

    # Remove markdown code blocks if present
    content = content.strip()
    if content.startswith("```"):
        # Remove ```json or ``` at start and ``` at end
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"{node_name}: Failed to parse JSON output: {e}")
        logger.debug(f"{node_name}: Raw content: {content[:500]}...")
        # Return a minimal valid structure
        return {
            "verdict": "FAIL",
            "skill_level": "beginner",
            "fatal_flaws": [],
            "strengths": [],
            "improvements": [],
            "next_topics": [],
            "_parse_error": str(e),
            "_raw_content": content
        }


def analyze_node(state: ReviewState, config: RunnableConfig):
    """Analyzes the submission against the model answer. Outputs structured JSON via with_structured_output."""
    logger.info("--- Analyze (Architecture) Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    submission = state["submission"]
    model_answer = state.get("model_answer", "N/A")
    original_code = state.get("original_code", "N/A")
    task_type = state["task_type"]

    lint_data = state.get("lint_result", {})
    lint_result = lint_data.get("content", "") if isinstance(lint_data, dict) else str(lint_data)

    research_data = state.get("research_result", {})
    research_result = research_data.get("content", "") if isinstance(research_data, dict) else str(research_data)

    logger.debug(f"Input - Lint Result: {lint_result}")

    # Switch prompt based on task type
    if task_type == "Programming":
        system_prompt = ARCHITECTURE_PROGRAMMING_SYSTEM
    else:
        system_prompt = ARCHITECTURE_PROMPT_ENGINEERING_SYSTEM

    # Add Research Instructions to System Prompt
    research_instruction = """
## Web検索の制御 (Dynamic Research)
分析を進める中で、外部知識、最新のライブラリ仕様、特定の技術用語の詳細調査が必要だと判断した場合は、`needs_research` を `true` に設定し、`research_queries` に検索クエリを記述してください。
リサーチ結果がある場合（入力の「リサーチ結果」が空でない場合）は、その情報も踏まえて最終的な判定を行ってください。リサーチ結果が十分であれば `needs_research` を `false` に戻してください。
"""
    system_prompt_with_research = system_prompt + research_instruction

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_with_research),
        ("user", """タスクタイプ: {task_type}

<submission>
{submission}
</submission>

<original_code>
{original_code}
</original_code>

<model_answer>
{model_answer}
</model_answer>

<lint_result>
{lint_result}
</lint_result>

<research_result>
{research_result}
</research_result>
""")
    ])

    # Use with_structured_output for reliable JSON schema enforcement (include_raw=True to get usage metadata)
    structured_llm = llm_analyze.with_structured_output(AnalyzeOutput, include_raw=True)
    chain = prompt | structured_llm

    inputs = {
        "submission": submission,
        "model_answer": model_answer,
        "original_code": original_code,
        "task_type": task_type,
        "lint_result": lint_result,
        "research_result": research_result
    }

    try:
        result = chain.invoke(inputs)
        analyze_result: AnalyzeOutput = result["parsed"]
        raw_message = result["raw"]
        analysis_json = analyze_result.model_dump()
        raw_content = json.dumps(analysis_json, ensure_ascii=False, indent=2)
        # Record token usage from raw message
        usage = record_usage(raw_message, "Analyze", "MODEL_ANALYZE", config=config, session_id=session_id)
    except Exception as e:
        logger.warning(f"Analyze: with_structured_output failed, falling back to manual parse: {e}")
        # Fallback to manual parsing with robust invocation
        chain_fallback = prompt | llm_analyze
        result = invoke_llm_with_fallback(
            chain=chain_fallback,
            inputs=inputs,
            node_name="Analyze",
            model_env_var="MODEL_ANALYZE",
            timeout=90,
            max_retries=3
        )
        # Handle different result types (AIMessage, list, str)
        if isinstance(result, BaseMessage):
            content_to_parse = result.content
        elif isinstance(result, list):
            # If result is a list, take first element
            content_to_parse = result[0].content if len(result) > 0 and isinstance(result[0], BaseMessage) else str(result)
        else:
            content_to_parse = result

        analysis_json = parse_json_output(content_to_parse, "Analyze")
        raw_content = content_to_parse if isinstance(content_to_parse, str) else str(content_to_parse)
        # Record usage from fallback
        usage = record_usage(result, "Analyze", "MODEL_ANALYZE", config=config, session_id=session_id)

    # Log structured output
    logger.info(f"Analyze: verdict={analysis_json.get('verdict')}, fatal_flaws={len(analysis_json.get('fatal_flaws', []))}")
    logger.debug(f"Output - Analysis Result (JSON): {json.dumps(analysis_json, ensure_ascii=False, indent=2)[:1000]}...")

    return {
        "analysis_result": {
            "content": raw_content,
            "json": analysis_json,
            "token_usage": usage
        },
        "status": "analyzed",
        "messages": [AIMessage(content=f"Analysis:\n{get_text_content(raw_content)}")]
    }

def fact_check_node(state: ReviewState, config: RunnableConfig):
    """Verifies the evidence in Analyze JSON output. Outputs JSON."""
    logger.info("--- Fact Check Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    analysis_data = state["analysis_result"]

    # Get both raw content and parsed JSON
    analysis_content = analysis_data.get("content", "") if isinstance(analysis_data, dict) else str(analysis_data)
    analysis_json = analysis_data.get("json", {}) if isinstance(analysis_data, dict) else {}

    submission = state["submission"]
    original_code = state.get("original_code", "N/A")

    fact_check_system_prompt = """あなたはエビデンス検証の専門家です。
Analyzeノードが出力したJSON内の主張を、提出物の実際の内容で検証してください。

## タスク

1. **evidence箇所の検証**: Analyze JSONのevidenceで指定された箇所に、本当にその内容があるか確認
2. **伝播の検証**: propagated_toで指定された箇所に、本当に伝播した情報があるか確認
3. **見落としの検出**: Analyzeが見落としている問題がないか確認

## 重要

- **楽観的すぎる評価をしない**: 「幸いにも上書きされた」のような評価は危険
- **全ての伝播を確認**: 数値だけでなく、日付・属性・その他の情報もチェック
- **具体的な引用で報告**: 検証結果には必ず実際のテキストを引用

## JSON出力形式

**必ずJSONのみを出力してください。**

```json
{{
  "verification_results": [
    {{
      "claim_path": "fatal_flaws[0]",
      "claim_summary": "AIが6,240人という数値を創作した",
      "evidence_location": "プロンプト1出力 > 3. 成果概要",
      "verified": true,
      "actual_content": "新規会員登録者数: 6,240人（実際に存在）",
      "note": null
    }},
    {{
      "claim_path": "fatal_flaws[0].evidence.propagated_to[0]",
      "claim_summary": "日付が伝播した",
      "evidence_location": "プロンプト2出力 > 2. 実施概要",
      "verified": true,
      "actual_content": "実施期間: 2024年3月1日 ～ 5月31日",
      "note": "元データにない具体的な日付が伝播"
    }}
  ],
  "missed_issues": [
    {{
      "type": "propagation_not_detected",
      "description": "顧客層の伝播が検出されていない",
      "evidence": {{
        "section": "プロンプト2出力 > 2. 実施概要",
        "quote": "対象顧客層: 20～40代のオンライン購買層",
        "note": "元データに記載なし"
      }}
    }}
  ],
  "false_claims": [],
  "overall_validity": true
}}
```"""

    # Include both raw content and JSON summary in the prompt (escaped for LangChain)
    if analysis_json:
        analysis_summary = f"分析JSON:\n{escape_for_prompt(json.dumps(analysis_json, ensure_ascii=False, indent=2))}"
    else:
        analysis_summary = f"分析内容:\n{escape_for_prompt(analysis_content)}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", fact_check_system_prompt),
        ("user", f"""<submission>
{{submission}}
</submission>

<original_code>
{{original_code}}
</original_code>

{analysis_summary}""")
    ])

    # Use with_structured_output for reliable JSON schema enforcement (include_raw=True to get usage metadata)
    structured_llm = llm_fact_check.with_structured_output(FactCheckOutput, include_raw=True)
    chain = prompt | structured_llm

    try:
        result = chain.invoke({"submission": submission, "original_code": original_code})
        fact_check_result: FactCheckOutput = result["parsed"]
        raw_message = result["raw"]
        fact_check_json = fact_check_result.model_dump()
        raw_content = json.dumps(fact_check_json, ensure_ascii=False, indent=2)
        # Record token usage from raw message
        usage = record_usage(raw_message, "Fact Check", "MODEL_FACT_CHECK", config=config, session_id=session_id)
    except Exception as e:
        logger.warning(f"Fact Check: with_structured_output failed, falling back to manual parse: {e}")
        chain_fallback = prompt | llm_fact_check
        result = chain_fallback.invoke({"submission": submission, "original_code": original_code})
        # Handle different result types
        if isinstance(result, BaseMessage):
            content_to_parse = result.content
        elif isinstance(result, list):
            content_to_parse = result[0].content if len(result) > 0 and isinstance(result[0], BaseMessage) else str(result)
        else:
            content_to_parse = result
        fact_check_json = parse_json_output(content_to_parse, "Fact Check")
        raw_content = content_to_parse if isinstance(content_to_parse, str) else str(content_to_parse)
        # Record usage from fallback
        usage = record_usage(result, "Fact Check", "MODEL_FACT_CHECK", config=config, session_id=session_id)

    logger.info(f"Fact Check: overall_validity={fact_check_json.get('overall_validity')}, missed_issues={len(fact_check_json.get('missed_issues', []))}")
    logger.debug(f"Output - Fact Check Result (JSON): {json.dumps(fact_check_json, ensure_ascii=False, indent=2)[:1000]}...")

    return {
        "fact_check_result": {
            "content": raw_content,
            "json": fact_check_json,
            "token_usage": usage
        },
        "messages": [AIMessage(content=f"Fact Check:\n{get_text_content(raw_content)}")]
    }

def draft_review_node(state: ReviewState, config: RunnableConfig):
    """Generates the initial review draft from JSON input. Outputs natural language."""
    logger.info("--- Draft Review Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    submission = state["submission"]

    # Get JSON data from Analyze and Fact Check
    analysis_data = state["analysis_result"]
    analysis_json = analysis_data.get("json", {}) if isinstance(analysis_data, dict) else {}
    analysis_content = analysis_data.get("content", "") if isinstance(analysis_data, dict) else str(analysis_data)

    fact_check_data = state["fact_check_result"]
    fact_check_json = fact_check_data.get("json", {}) if isinstance(fact_check_data, dict) else {}
    fact_check_content = fact_check_data.get("content", "") if isinstance(fact_check_data, dict) else str(fact_check_data)

    # Format JSON for prompt (escaped for LangChain)
    analyze_json_str = escape_for_prompt(json.dumps(analysis_json, ensure_ascii=False, indent=2)) if analysis_json else escape_for_prompt(analysis_content)
    fact_check_json_str = escape_for_prompt(json.dumps(fact_check_json, ensure_ascii=False, indent=2)) if fact_check_json else escape_for_prompt(fact_check_content)

    # Build prompt with escaped JSON inline (not as variables)
    user_message = f"""提出物:
{{submission}}

## Analyze JSON
```json
{analyze_json_str}
```

## Fact Check JSON
```json
{fact_check_json_str}
```

上記のJSON構造化データを、生徒向けの読みやすいレビューコメントに変換してください。
特に、evidenceで指定された引用と、propagated_toで示された伝播先は必ずレビューに含めてください。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", DRAFT_REVIEW_SYSTEM_PROMPT),
        ("user", user_message)
    ])
    chain = prompt | llm_draft

    logger.info("Draft Review: Invoking LLM...")
    # Use robust invocation with timeout and fallback
    result = invoke_llm_with_fallback(
        chain=chain,
        inputs={
            "submission": submission
        },
        node_name="Draft Review",
        model_env_var="MODEL_DRAFT",
        timeout=90,
        max_retries=3
    )

    logger.info("Draft Review: LLM invocation complete. Recording usage...")
    usage = record_usage(result, "Draft Review", "MODEL_DRAFT", config=config, session_id=session_id)
    if usage["total"] == 0:
         logger.warning("Draft Review: Token usage came back as 0!")
    else:
         logger.info(f"Draft Review: Token usage recorded: {usage}")

    logger.info(f"Draft Review Output:\n{result.content}")
    logger.debug(f"Draft Review Output (full): {len(result.content)} chars")

    # Get verdict from Analyze JSON
    json_verdict = analysis_json.get("verdict", "FAIL")

    return {
        "draft_review": {
            "content": get_text_content(result.content),
            "json_verdict": json_verdict,  # Track the JSON verdict for Critique
            "token_usage": usage
        },
        "status": "drafted",
        "messages": [AIMessage(content=f"Draft Review:\n{get_text_content(result.content)}")]
    }

def critique_node(state: ReviewState, config: RunnableConfig):
    """Checks JSON↔Draft consistency, Severity Mismatch, and tone. Outputs structured JSON via with_structured_output."""
    logger.info("--- Critique Node Entered ---")
    session_id = config["configurable"].get("db_session_id")

    # Get Draft (natural language)
    draft_data = state["draft_review"]
    draft = draft_data.get("content", "") if isinstance(draft_data, dict) else str(draft_data)
    json_verdict = draft_data.get("json_verdict", "FAIL") if isinstance(draft_data, dict) else "FAIL"

    submission = state["submission"]

    # Get Analyze JSON
    analysis_data = state["analysis_result"]
    analysis_json = analysis_data.get("json", {}) if isinstance(analysis_data, dict) else {}
    analysis_content = analysis_data.get("content", "") if isinstance(analysis_data, dict) else str(analysis_data)

    # Format inputs for prompt (escaped for LangChain)
    analyze_json_str = escape_for_prompt(json.dumps(analysis_json, ensure_ascii=False, indent=2)) if analysis_json else escape_for_prompt(analysis_content)
    submission_escaped = escape_for_prompt(submission)
    draft_escaped = escape_for_prompt(draft)

    user_message = f"""## 提出物
{submission_escaped}

## Analyze JSON（構造化分析データ）
```json
{analyze_json_str}
```

## Draft（自然言語レビュー）
{draft_escaped}

---

上記のAnalyze JSONとDraftを比較し、以下をJSONでチェックしてください：
1. Severity Mismatch: fatal_flawsがあるのにDraftがPASSになっていないか
2. JSON↔Draft整合性: JSONの指摘がDraftに含まれているか（特にpropagated_to）
3. トーン: 馴れ馴れしい、上から目線、子供扱いなどがないか"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIQUE_SYSTEM_PROMPT),
        ("user", user_message)
    ])

    # Use with_structured_output for reliable JSON schema enforcement (include_raw=True to get usage metadata)
    structured_llm = llm_critique.with_structured_output(CritiqueOutput, include_raw=True)
    chain = prompt | structured_llm

    inputs = {}

    try:
        result = chain.invoke(inputs)
        critique_result: CritiqueOutput = result["parsed"]
        raw_message = result["raw"]
        critique_json = critique_result.model_dump()
        raw_content = json.dumps(critique_json, ensure_ascii=False, indent=2)
        # Record token usage from raw message
        usage = record_usage(raw_message, "Critique", "MODEL_CRITIQUE", config=config, session_id=session_id)
    except Exception as e:
        logger.warning(f"Critique: with_structured_output failed, falling back to manual parse: {e}")
        chain_fallback = prompt | llm_critique
        result = invoke_llm_with_fallback(
            chain=chain_fallback,
            inputs=inputs,
            node_name="Critique",
            model_env_var="MODEL_CRITIQUE",
            timeout=90,
            max_retries=3
        )
        # Handle different result types (AIMessage, list, str)
        if isinstance(result, BaseMessage):
            content_to_parse = result.content
        elif isinstance(result, list):
            content_to_parse = result[0].content if len(result) > 0 and isinstance(result[0], BaseMessage) else str(result)
        else:
            content_to_parse = result
        critique_json = parse_json_output(content_to_parse, "Critique")
        raw_content = content_to_parse if isinstance(content_to_parse, str) else str(content_to_parse)
        # Record usage from fallback
        usage = record_usage(result, "Critique", "MODEL_CRITIQUE", config=config, session_id=session_id)

    # Log critique results
    approve = critique_json.get("overall", {}).get("approve", False)
    has_mismatch = critique_json.get("severity_mismatch_check", {}).get("mismatch", False)
    is_too_lenient = critique_json.get("strictness_check", {}).get("is_draft_too_lenient", False)
    leniency_count = len(critique_json.get("strictness_check", {}).get("leniency_issues", []))
    accepted_persuasions = critique_json.get("overall", {}).get("accepted_persuasions", [])
    accepted_count = len([p for p in accepted_persuasions if p.get("accepted", False)])
    rejected_count = len([p for p in accepted_persuasions if not p.get("accepted", True)])
    logger.info(f"Critique: approve={approve}, severity_mismatch={has_mismatch}, too_lenient={is_too_lenient}, leniency_issues={leniency_count}, persuasions_accepted={accepted_count}, persuasions_rejected={rejected_count}")
    logger.debug(f"Output - Critique (JSON): {json.dumps(critique_json, ensure_ascii=False, indent=2)[:1000]}...")

    return {
        "critique_comment": {
            "content": raw_content,
            "json": critique_json,
            "token_usage": usage
        },
        "messages": [AIMessage(content=f"Critique:\n{get_text_content(raw_content)}")]
    }

def refine_review_node(state: ReviewState, config: RunnableConfig):
    """Refines the review based on Critique JSON or human feedback."""
    logger.info("--- Refine Review Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    current_draft_data = state["draft_review"]
    current_draft = current_draft_data.get("content", "") if isinstance(current_draft_data, dict) else str(current_draft_data)

    feedback = state.get("human_feedback")

    # Get Critique JSON
    critique_data = state.get("critique_comment")
    critique_json = critique_data.get("json", {}) if isinstance(critique_data, dict) else {}
    critique_content = critique_data.get("content", "") if isinstance(critique_data, dict) else str(critique_data) if critique_data else None
    submission = state["submission"]

    # Get Analyze JSON for reference
    analysis_data = state["analysis_result"]
    analysis_json = analysis_data.get("json", {}) if isinstance(analysis_data, dict) else {}

    # Build revision instructions from Critique JSON
    instructions = ""
    if feedback:
        instructions += f"## ユーザーフィードバック\n{feedback}\n\n"

    if critique_json:
        # Extract revisions from Critique JSON
        revisions = critique_json.get("overall", {}).get("revisions", [])
        if revisions:
            instructions += "## Critique JSONからの修正指示\n"
            for r in revisions:
                priority = r.get("priority", "medium")
                rev_type = r.get("type", "unknown")
                instruction = r.get("instruction", "")
                instructions += f"- **[{priority}] {rev_type}**: {instruction}\n"
            instructions += "\n"

        # Add missing_in_draft items
        missing = critique_json.get("json_draft_consistency", {}).get("missing_in_draft", [])
        if missing:
            instructions += "## Draftに不足している項目（Analyze JSONから追加が必要）\n"
            for m in missing:
                json_path = m.get("json_path", "")
                content = m.get("content", "")
                instructions += f"- `{json_path}`: {content}\n"
            instructions += "\n"

        # Severity Mismatch warning
        if critique_json.get("severity_mismatch_check", {}).get("mismatch"):
            instructions += "## 🚨 Severity Mismatch 警告\n"
            instructions += "致命的欠陥（fatal_flaws）があるにもかかわらずPASS判定になっています。\n"
            instructions += "結論を**FAIL**に変更し、致命的欠陥の説明を追加してください。\n\n"
    elif critique_content:
        instructions += f"## 品質管理レビュー（QA）\n{critique_content}\n"

    if not instructions:
        logger.info("No refinement needed")
        # IMPORTANT: Preserve draft_review state even when no changes are made
        return {
            "draft_review": current_draft_data,  # Keep existing draft
            "status": "refined"
        }

    # Format inputs for prompt (escaped for LangChain)
    analyze_json_str = escape_for_prompt(json.dumps(analysis_json, ensure_ascii=False, indent=2)) if analysis_json else ""
    submission_escaped = escape_for_prompt(submission)
    current_draft_escaped = escape_for_prompt(current_draft)
    instructions_escaped = escape_for_prompt(instructions)

    user_message = f"""## 提出物
{submission_escaped}

## 参考: Analyze JSON
```json
{analyze_json_str}
```

## 現在の下書き
{current_draft_escaped}

## 修正指示
{instructions_escaped}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはプロフェッショナルなAIエンジニア兼「調停者」です。
Critique（品質監査官）の指摘を受け、レビュー下書きを修正・改善してください。

## あなたの役割

Critiqueは「冷徹な品質監査官」として過剰に厳格な指摘をすることがあります。
あなたは以下の2つのバランスを取る調停者です：

1. **厳格さの維持**: Critiqueの正当な指摘は尊重し、修正を行う
2. **教育的配慮の保持**: 学習者のモチベーションを完全に無視してはならない

## 🔴 絶対に妥協してはならない指摘（Critical）

以下の指摘は**必ず**修正してください。説得は不可です：

- **Severity Mismatch**: fatal_flawsがあるのにPASS → 必ずFAILに変更
- **セキュリティリスクの軽視**: Prompt Injection、Data Leakage等の軽視 → 必ず強調
- **ハルシネーション誘発の軽視**: 虚偽情報生成リスクの軽視 → 必ず強調

## 🟠 説得可能な指摘（High/Medium）

以下の指摘については、**正当な理由**があれば部分的に反論できます：

### 「FAIL判定で良い点を先に挙げている」への反論例
「学習者のモチベーション維持のため、良い点を先に認識させることで、その後の致命的欠陥の指摘を建設的に受け止めやすくしています。ただし、致命的欠陥のセクションを強化し、FAILの理由を明確にしました。」

### 「温情的表現がある」への反論例
「"残念ながら"は削除しましたが、"今後の学習に活かしてください"は、具体的な学習トピックと共に残しました。これは励ましではなく、建設的な次のステップの提示です。」

## 出力形式

レビュー下書きを修正し、**末尾に「Critiqueへの応答」セクション**を追加してください：

```
（修正されたレビュー本文）

---
## 📝 Critiqueへの応答（内部用・最終出力には含まれない）

### 対応した指摘
- [priority] type: 対応内容

### 説得・妥協した指摘
- [priority] type: 「元の指摘」→「対応」
  - 理由: なぜこの対応が適切か
```

## 重要ルール

- Analyze JSONのfatal_flawsに記載されている問題は、必ずレビューに含める
- evidence.propagated_toで指定された伝播先は、具体的な引用と共に言及
- トーンは「プロフェッショナルで事実ベース」を基本とし、過度な励ましは避ける
- ただし、「学習者のモチベーション」と「事実ベースの厳格さ」の両立を目指す"""),
        ("user", user_message)
    ])
    chain = prompt | llm_refine

    # Use robust invocation with timeout and fallback
    result = invoke_llm_with_fallback(
        chain=chain,
        inputs={},
        node_name="Refine Review",
        model_env_var="MODEL_REFINE",
        timeout=90,
        max_retries=3
    )

    usage = record_usage(result, "Refine Review", "MODEL_REFINE", config=config, session_id=session_id)
    logger.info(f"Refined Review Output:\n{result.content}")
    logger.debug(f"Refined Review Output (full): {len(result.content)} chars")

    # Increment refine counter
    current_refine_count = state.get("refine_count", 0) + 1
    logger.info(f"Refine iteration: {current_refine_count}/{MAX_REFINE_ITERATIONS}")

    return {
        "draft_review": {"content": get_text_content(result.content), "token_usage": usage},
        "status": "refined",
        "critique_comment": None, # Clear critique after applying
        "refine_count": current_refine_count,  # Track loop iterations
        "messages": [AIMessage(content=f"Refined Review:\n{get_text_content(result.content)}")]
    }

def consistency_check_node(state: ReviewState, config: RunnableConfig):
    """Checks for contradictions in the final review."""
    logger.info("--- Consistency Check Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    review_data = state["draft_review"]
    review = review_data.get("content", "") if isinstance(review_data, dict) else str(review_data)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは品質保証スペシャリストです。レビュー内容に論理的矛盾やトーンの不一致がないか確認してください。\n問題がなければ'PASS'、矛盾があれば指摘してください。"),
        ("user", "レビュー:\n{review}")
    ])
    chain = prompt | llm_consistency
    result = chain.invoke({"review": review})

    usage = record_usage(result, "Consistency Check", "MODEL_CONSISTENCY", config=config, session_id=session_id)
    logger.info(f"Consistency Check Output:\n{result.content}")
    logger.debug(f"Consistency Check Output (full): {len(result.content)} chars")

    return {
        "consistency_result": {"content": get_text_content(result.content), "token_usage": usage},
        "messages": [AIMessage(content=f"Consistency Check:\n{get_text_content(result.content)}")]
    }

def evaluate_node(state: ReviewState, config: RunnableConfig):
    """LLM-as-a-Judge: Evaluates the final review quality."""
    logger.info("--- Evaluate Node Entered ---")
    session_id = config["configurable"].get("db_session_id")
    review_data = state["draft_review"]
    review = review_data.get("content", "") if isinstance(review_data, dict) else str(review_data)
    submission = state["submission"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは公平な裁判官です。レビューの品質を「厳格さ(Strictness)」と「教育的価値(Educational Value)」の2点で1-5段階評価してください。\nJSON形式のみで返してください。例: {{'strictness': 4, 'educational': 5}}"),
        ("user", "提出物:\n{submission}\n\nレビュー:\n{review}")
    ])
    chain = prompt | llm_judge
    result = chain.invoke({"submission": submission, "review": review})

    usage = record_usage(result, "Evaluate", "MODEL_JUDGE", config=config, session_id=session_id)

    # Simple parsing (robustness improvement needed for prod)
    import json
    try:
        # Extract JSON from markdown block if present
        content = get_text_content(result.content)

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        eval_dict = json.loads(content.strip())
    except Exception as e:
        logger.error(f"Evaluation parsing failed: {e}")
        eval_dict = {"strictness": 0, "educational": 0, "error": "Failed to parse"}

    logger.info("Evaluation Completed")
    logger.info(f"Evaluation Output (raw):\n{result.content}")
    logger.info(f"Evaluation Output (parsed): {json.dumps(eval_dict, ensure_ascii=False, indent=2)}")
    logger.debug(f"Evaluation Output (full): {len(result.content)} chars")

    return {
        "evaluation_result": {"content": eval_dict, "token_usage": usage},
        "messages": [AIMessage(content=f"Evaluation:\n{get_text_content(result.content)}")]
    }

def scoring_node(state: ReviewState, config: RunnableConfig):
    """Scores the submission on a 0-100 scale based on complexity, correctness, quality, sophistication, and improvement."""
    logger.info("--- Scoring Node Entered ---")
    session_id = config["configurable"].get("db_session_id")

    # Extract data from state
    submission = state["submission"]
    review_data = state.get("draft_review")
    review = review_data.get("content", "") if isinstance(review_data, dict) else str(review_data) if review_data else ""

    evaluation_data = state.get("evaluation_result")
    evaluation = evaluation_data.get("content", "") if isinstance(evaluation_data, dict) else str(evaluation_data) if evaluation_data else ""

    analysis_data = state.get("analysis_result")
    analysis = analysis_data.get("content", "") if isinstance(analysis_data, dict) else str(analysis_data) if analysis_data else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SCORING_SYSTEM_PROMPT),
        ("user", """提出物:
{submission}

AI評価レポート:
{evaluation}

分析結果（参考）:
{analysis}

最終レビュー:
{review}

上記の情報を基に、0-100点のスコアを付与してください。""")
    ])

    chain = prompt | llm_judge
    result = chain.invoke({
        "submission": submission,
        "evaluation": str(evaluation),
        "analysis": str(analysis),
        "review": review
    })

    usage = record_usage(result, "Scoring", "MODEL_JUDGE", config=config, session_id=session_id)

    # Parse JSON response
    import json
    try:
        content = get_text_content(result.content)

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        score_dict = json.loads(content.strip())

        # Extract score (ensure it's 0-100)
        score = score_dict.get("score", 0)
        if not isinstance(score, int):
            score = int(float(score))
        score = max(0, min(100, score))  # Clamp to 0-100

        logger.info(f"Scoring Completed: {score}/100")
        logger.info(f"Scoring Output (raw):\n{result.content}")
        logger.info(f"Scoring Output (parsed): {json.dumps(score_dict, ensure_ascii=False, indent=2)}")
        logger.debug(f"Scoring Output (full): {len(result.content)} chars")

    except Exception as e:
        logger.error(f"Scoring parsing failed: {e}")
        score = 0
        score_dict = {"score": 0, "reason": f"Failed to parse: {e}", "breakdown": {}}

    # Update session status to completed
    if session_id:
        try:
            db = next(get_db())
            session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
            if session:
                session.status = "completed"
                db.commit()
                logger.info(f"Session {session_id} status updated to 'completed'")
            db.close()
        except Exception as e:
            logger.error(f"Failed to update session status: {e}")

    return {
        "score": score,
        "status": "completed",
        "messages": [AIMessage(content=f"Score: {score}/100\n{get_text_content(result.content)}")]
    }

# --- Graph Definition ---

workflow = StateGraph(ReviewState)

# Add Nodes
workflow.add_node("router", router_node) # New Entry Point
workflow.add_node("cli_bypass", cli_bypass_node) # CLI Bypass
workflow.add_node("manager", manager_node) # Manager Node
workflow.add_node("lint", lint_check_node)
workflow.add_node("research", research_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("fact_check", fact_check_node)
workflow.add_node("draft", draft_review_node)
workflow.add_node("critique", critique_node)
workflow.add_node("refine", refine_review_node)
workflow.add_node("consistency", consistency_check_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("scoring", scoring_node)

# Add Edges
workflow.set_entry_point("router") # Start with Router

def route_after_router(state: ReviewState):
    """Decides next step based on CLI bypass flag."""
    return "lint"

def route_after_research(state: ReviewState):
    review_mode = state.get("review_mode", "standard")
    use_cli_bypass = state.get("use_cli_bypass", False)

    # TEMPORARY FIX: Disable CLI bypass due to massive log output issue
    # Always use standard analyze flow
    return "analyze"

    # # Backward compatibility (DISABLED)
    # if use_cli_bypass and review_mode == "standard":
    #     review_mode = "hybrid"
    #
    # if review_mode == "consensus":
    #     return ["analyze", "cli_bypass"] # Parallel execution
    # elif review_mode == "hybrid":
    #     return "cli_bypass"
    # else:
    #     return "analyze"

workflow.add_conditional_edges(
    "router",
    route_after_router,
    {
        "lint": "lint"
    }
)

# After Lint, always go to Analyze
workflow.add_edge("lint", "analyze")

def should_research(state: ReviewState):
    """Determines if we need to run research based on Analyze output."""
    analysis_data = state.get("analysis_result", {})
    analysis_json = analysis_data.get("json", {}) if isinstance(analysis_data, dict) else {}

    needs_research = analysis_json.get("needs_research", False)
    research_data = state.get("research_result", {})
    already_done = research_data.get("done", False) if isinstance(research_data, dict) else False
    disable_web_search = state.get("disable_web_search", False)

    if needs_research and not already_done and not disable_web_search:
        logger.info("[ROUTING] Dynamic Research needed. Moving to research node.")
        return "research"
    else:
        logger.info("[ROUTING] No research needed (or already done/disabled). Moving to fact_check.")
        return "fact_check"

workflow.add_conditional_edges(
    "analyze",
    should_research,
    {
        "research": "research",
        "fact_check": "fact_check"
    }
)

# After Research, go back to Analyze for refinement
workflow.add_edge("research", "analyze")

# Standard Flow Continuation
workflow.add_edge("fact_check", "draft")
workflow.add_edge("draft", "critique")

# CLI Hybrid Flow
workflow.add_edge("cli_bypass", "manager")
workflow.add_edge("manager", "critique") # Join back at Critique

workflow.add_edge("critique", "refine") # Auto-refine after critique
# After refine, we pause for human check.

def should_continue(state: ReviewState):
    refine_count = state.get("refine_count", 0)

    # Check if we've hit the refine iteration limit
    if refine_count >= MAX_REFINE_ITERATIONS:
        logger.warning(f"⚠️ Max refine iterations ({MAX_REFINE_ITERATIONS}) reached. Forcing consistency check.")
        return "consistency"

    # If human feedback is present, go back to refine
    if state.get("human_feedback"):
        return "refine"

    # Otherwise, go to consistency check
    return "consistency"

workflow.add_conditional_edges(
    "refine",
    should_continue,
    {
        "refine": "refine",
        "consistency": "consistency"
    }
)

workflow.add_edge("consistency", "evaluate")
workflow.add_edge("evaluate", "scoring")
workflow.add_edge("scoring", END)

# Compile - Use SqliteSaver for persistence across server restarts
#
# NOTE: For production with multiple servers/processes, consider migrating to PostgresSaver:
#   1. Install: pip install langgraph-checkpoint-postgres
#   2. Replace the code below with:
#      from langgraph.checkpoint.postgres import PostgresSaver
#      import os
#      # Get connection string from environment variable
#      postgres_conn_string = os.getenv("POSTGRES_CHECKPOINT_URL", "postgresql://user:pass@localhost/dbname")
#      memory = PostgresSaver.from_conn_string(postgres_conn_string)
#   3. Benefits: Multi-server support, better concurrency, production-ready
#   4. Migration: Existing SqliteSaver checkpoints can be migrated via data export/import
#
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Use persistent SQLite checkpoint storage
# Note: SqliteSaver is suitable for single-server deployments.
# For multi-server/production, use PostgresSaver (see comment above).
checkpoint_db_path = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")
# Note: check_same_thread=False is OK as SqliteSaver uses a lock for thread safety
checkpoint_conn = sqlite3.connect(checkpoint_db_path, check_same_thread=False)
memory = SqliteSaver(checkpoint_conn)

logger.info(f"Using SqliteSaver for checkpoints: {checkpoint_db_path}")

# Interrupt after fact_check (Analysis check) and refine (Draft check)
app = workflow.compile(checkpointer=memory, interrupt_after=["fact_check", "refine"])


