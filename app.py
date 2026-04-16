import streamlit as st
import uuid
import os
import logging
from dotenv import load_dotenv

# Load env vars first!
load_dotenv()

from graph import app as graph_app, validate_input, CREDIT_PRICE_USD, MAX_INPUT_TOKENS, PRICING_TIERS, _BILLING_ENABLED
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from database import get_db, ReviewSession, ReviewMessage, ReviewResult, ReviewStepLog, init_db
from experiments.dspy_module import generate_review_dspy
from services import (
    extract_displayable_content,
    save_step_log,
    run_database_migrations,
    get_session_state_from_db,
    get_score_display_info,
    get_status_badge,
    to_json_str,
    reconstruct_langgraph_state_from_db
)
import json

def render_timeline(logs):
    """Renders the review process timeline from logs."""
    for log in logs:
        with st.expander(f"{log.step_name} ({log.created_at.strftime('%H:%M:%S')})", expanded=False):
            try:
                data = json.loads(log.output_data)
                if isinstance(data, dict) and "content" in data:
                    content = data["content"]

                    # If json key exists, content is usually text (for Draft Review, etc.)
                    # and json contains the structured data
                    if "json" in data and isinstance(data["json"], dict):
                        # Display main content as text (for Draft Review, Refined Review, etc.)
                        content_text = extract_displayable_content(content)
                        st.markdown(content_text)

                        # Display structured JSON data separately
                        st.divider()
                        st.caption("📋 Structured JSON Data")
                        st.json(data["json"])
                    else:
                        # No json key - check if content itself is JSON (for Analyze, Fact Check, etc.)
                        is_json_content = False
                        json_data = None

                        # Try to parse as JSON
                        if isinstance(content, str):
                            try:
                                json_data = json.loads(content)
                                # Only treat as JSON if it's actually a JSON object/array
                                # and not just a string that happens to be valid JSON
                                if isinstance(json_data, (dict, list)):
                                    is_json_content = True
                            except (json.JSONDecodeError, TypeError):
                                pass
                        elif isinstance(content, dict):
                            json_data = content
                            is_json_content = True

                        # Display JSON with proper formatting
                        if is_json_content:
                            st.json(json_data)
                        else:
                            # Main Content - Use helper function to properly decode
                            content_text = extract_displayable_content(content)
                            st.markdown(content_text)

                    # Metadata Section
                    st.divider()
                    st.caption("📊 Process Metadata")

                    # Token Usage
                    if "token_usage" in data:
                        usage = data["token_usage"]
                        st.markdown(f"**Token Usage**: Input: `{usage.get('input')}` / Output: `{usage.get('output')}` / Total: `{usage.get('total')}`")

                    # Research Specifics
                    if "queries" in data:
                        st.markdown("**🔍 Generated Queries:**")
                        for q in data["queries"]:
                            st.code(q, language="text")
            except Exception as e:
                st.error(f"Error rendering log for {log.step_name}: {e}")

# Setup logger
logger = logging.getLogger("streamlit_app")

# Page Config
st.set_page_config(page_title="Teacher's Copilot", layout="wide")

# Initialize DB
if "db_initialized" not in st.session_state:
    init_db()
    # Run migrations to add missing columns if needed
    try:
        run_database_migrations()
    except Exception as e:
        st.warning(f"Database migration warning: {e}")
    st.session_state.db_initialized = True

# Session State Init
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_state" not in st.session_state:
    st.session_state.current_state = None # To track graph state

# --- Sidebar ---
with st.sidebar:
    st.title("Teacher's Copilot")
    st.markdown("AI-Assisted Student Review System")

    # [NEW] Display Prompt Configuration
    import prompts  # Dynamic load check
    lang_flag = "🇺🇸" if prompts.get_prompt_language() == "en" else "🇯🇵"
    tier_label = "🔐 Enterprise" if prompts.get_prompt_version() == "enterprise" else "📦 Lite (OSS)"

    st.markdown("### ⚙️ System Config")
    st.info(f"""
    **Prompt Set**:
    {lang_flag} {prompts.get_prompt_language().upper()} | {tier_label}
    """)
    # --------------------------------

    task_type = st.selectbox("Task Type", ["Prompt Engineering", "Programming"])

    # Review Mode Selection
    mode_options = {
        "Standard (Graph Agent)": "standard",
        "Hybrid (Cost-Optimized)": "hybrid",
        "Consensus (Quality-Optimized)": "consensus"
    }
    selected_mode_label = st.radio(
        "Review Mode",
        options=list(mode_options.keys()),
        index=1, # Default to Consensus
        help="Standard: Pure Graph Agent.\nHybrid: Uses CLI for reasoning (Cheaper).\nConsensus: Merges Graph & CLI (Best Quality)."
    )
    review_mode = mode_options[selected_mode_label]

    # Legacy flag support
    use_cli_bypass = (review_mode in ["hybrid", "consensus"])

    # Web Search Disable Option
    disable_web_search = st.checkbox(
        "Disable Web Search",
        value=True,
        help="Recommended if web search has high failure rate."
    )

    st.divider()

    if st.button("New Review Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.current_state = None
        if "db_session_id" in st.session_state:
            del st.session_state.db_session_id
        if "dspy_result" in st.session_state:
            del st.session_state.dspy_result
        if "result_saved" in st.session_state:
            del st.session_state.result_saved
        if "selected_history_id" in st.session_state:
            del st.session_state.selected_history_id
        if "starting_review" in st.session_state:
            del st.session_state.starting_review
        st.rerun()

    st.divider()

    # Review History Section
    st.subheader("📚 Review History")

    db = next(get_db())
    # Get latest 20 review sessions, ordered by creation date (newest first)
    past_sessions = db.query(ReviewSession).order_by(ReviewSession.created_at.desc()).limit(20).all()
    db.close()

    if past_sessions:
        for session in past_sessions:
            # Format datetime for display
            created_time = session.created_at.strftime("%m/%d %H:%M")
            # Truncate submission for preview
            submission_preview = session.submission_content[:50] + "..." if len(session.submission_content) > 50 else session.submission_content

            # Get score if available
            db_for_score = next(get_db())
            result = db_for_score.query(ReviewResult).filter(ReviewResult.session_id == session.id).first()
            score_display = ""
            if result and result.score is not None:
                score_display = f" | スコア: {result.score}/100"
            db_for_score.close()

            with st.expander(f"#{session.id} - {created_time} ({session.task_type}){score_display}", expanded=False):
                st.caption(f"提出物: {submission_preview}")
                if result and result.score is not None:
                    st.metric("Score", f"{result.score}/100")
                if st.button(f"View This Review", key=f"load_session_{session.id}"):
                    # Set the selected session ID AND thread_id for LangGraph checkpoint lookup
                    st.session_state.selected_history_id = session.id
                    st.session_state.thread_id = session.id  # ✅ Sync thread_id for checkpoint lookup
                    st.session_state.db_session_id = session.id  # ✅ Set db_session_id for resume
                    st.rerun()
    else:
        st.info("No review history found")

    st.divider()

# --- Main Content ---
# New Review Page
# 1. Input Section (Always visible at top)
with st.container():
    st.subheader("📝 Student Submission")

    # Log current state on every render
    render_log = f"[RENDER] Page render. thread_id={st.session_state.get('thread_id')}, db_session_id={st.session_state.get('db_session_id')}, starting_review={st.session_state.get('starting_review')}"
    print(render_log)  # Force stdout
    logger.info(render_log)

    # If session already started OR starting, hide input form completely
    has_session = "db_session_id" in st.session_state
    is_starting = st.session_state.get("starting_review", False)

    state_log = f"[RENDER] has_session={has_session}, is_starting={is_starting}"
    print(state_log)
    logger.info(state_log)

    # Create button placeholder outside if/else to control visibility dynamically
    button_placeholder = st.empty()

    if has_session or is_starting:
        hide_log = "[RENDER] Session active - hiding input form"
        print(hide_log)
        logger.info(hide_log)
        # Clear button placeholder to hide any lingering buttons
        button_placeholder.empty()
        with st.expander("Show Submission & Model Answer", expanded=False):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                st.text_area("Submission", value=st.session_state.get("submission_content", ""), height=300, disabled=True, key="sub_display")
            with col_in2:
                st.text_area("Model Answer", value=st.session_state.get("model_answer_content", ""), height=300, disabled=True, key="model_display")
        if is_starting and not has_session:
            st.warning("🔄 Starting review... Please do not refresh the page.")
    else:
        logger.info("[RENDER] No session - showing input form and button")
        submission = st.text_area("Paste submission here...", height=300, key="submission_input")
        original_code = st.text_area("Original Code (Optional - for Diff tasks)", height=150, key="original_code_input")
        model_answer = st.text_area("Model Answer (Optional)", height=150, key="model_answer_input")

        # --- Pricing Calculator (Server-side calculation with debounce) ---
        import time
        import hashlib

        if submission:
            # Create hash of ALL content to detect changes
            content_str = (submission or "") + (original_code or "") + (model_answer or "")
            content_hash = hashlib.md5(content_str.encode()).hexdigest()[:8]

            # Check if we need to recalculate (content changed)
            if st.session_state.get("last_content_hash") != content_hash:
                # Show calculating spinner
                with st.spinner("💰 Calculating cost..."):
                    time.sleep(0.5)  # Brief delay for UX (debounce-like effect)
                    # Server-side calculation (single source of truth)
                    pricing_info = validate_input(submission, original_code=original_code, model_answer=model_answer)
                    st.session_state.last_content_hash = content_hash
                    st.session_state.cached_pricing = pricing_info
            else:
                # Use cached pricing
                pricing_info = st.session_state.get("cached_pricing", validate_input(submission, original_code=original_code, model_answer=model_answer))

            # Display validation / pricing info
            if pricing_info["valid"]:
                if _BILLING_ENABLED:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                padding: 1rem; border-radius: 10px; border: 1px solid #0f3460;">
                        <h4 style="margin: 0; color: #e94560;">Cost Estimate (Server-side)</h4>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <div>
                                <span style="color: #a0a0a0;">Tokens:</span>
                                <strong style="color: #fff;">{pricing_info['tokens']:,}</strong>
                            </div>
                            <div>
                                <span style="color: #a0a0a0;">Tier:</span>
                                <strong style="color: #e94560;">{pricing_info['tier']}</strong>
                            </div>
                            <div>
                                <span style="color: #a0a0a0;">Cost:</span>
                                <strong style="color: #00d9ff;">{pricing_info['credits']} credits (${pricing_info['cost_usd']:.2f})</strong>
                            </div>
                        </div>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #666;">
                            Final billing is determined by the server at execution time.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                                padding: 1rem; border-radius: 10px; border: 1px solid #0f3460;">
                        <h4 style="margin: 0; color: #e94560;">Input Estimate</h4>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                            <div>
                                <span style="color: #a0a0a0;">Tokens:</span>
                                <strong style="color: #fff;">{pricing_info['tokens']:,}</strong>
                            </div>
                            <div>
                                <span style="color: #a0a0a0;">Limit:</span>
                                <strong style="color: #00d9ff;">{MAX_INPUT_TOKENS:,}</strong>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(pricing_info["message"])
        else:
            # Clear cache when no input
            if "last_content_hash" in st.session_state:
                del st.session_state.last_content_hash
            if "cached_pricing" in st.session_state:
                del st.session_state.cached_pricing

            # Show info when no input
            if _BILLING_ENABLED:
                st.info(f"""
                **Pricing** (1 credit = ${CREDIT_PRICE_USD})
                | Tier | Tokens | Cost |
                |------|--------|------|
                | Small | ~5,000 | 1 credit (${CREDIT_PRICE_USD * 1:.0f}) |
                | Medium | ~20,000 | 2 credits (${CREDIT_PRICE_USD * 2:.0f}) |
                | Large | ~50,000 | 4 credits (${CREDIT_PRICE_USD * 4:.0f}) |
                | Enterprise | ~100,000 | 8 credits (${CREDIT_PRICE_USD * 8:.0f}) |
                """)
            else:
                st.info(f"""
                **Input Limit**

                Maximum input: {MAX_INPUT_TOKENS:,} tokens

                For Enterprise pricing, please contact us.
                """)

        # Button control logic - ONLY shown when no session exists
        logger.info("[RENDER] Preparing Start Review button")
        cached_pricing = st.session_state.get("cached_pricing")
        has_session_check = "db_session_id" not in st.session_state
        not_starting = not st.session_state.get("starting_review", False)
        can_start = (
            submission and
            cached_pricing and
            cached_pricing.get("valid", False) and
            has_session_check and
            not_starting
        )
        logger.info(f"[RENDER] can_start={can_start} (has_session_check={has_session_check}, not_starting={not_starting})")

        # Callback to set flag BEFORE page rerun (with cooldown protection)
        import time as _time
        COOLDOWN_SECONDS = 2

        def on_start_review():
            # FIRST: Check if session already exists - abort immediately
            if st.session_state.get("db_session_id") or st.session_state.get("starting_review"):
                print(f"[BUTTON] ABORT: Session already exists (db_session_id={st.session_state.get('db_session_id')}, starting_review={st.session_state.get('starting_review')})")
                logger.warning(f"[BUTTON] ABORT: Session already exists")
                return

            # Cooldown check: ignore clicks within 2 seconds
            last_click = st.session_state.get("last_start_click", 0)
            now = _time.time()

            logger.info(f"[BUTTON] on_start_review called. last_click={last_click}, now={now}, diff={now - last_click:.2f}s")
            logger.info(f"[BUTTON] Current state: starting_review={st.session_state.get('starting_review')}, db_session_id={st.session_state.get('db_session_id')}")

            if now - last_click < COOLDOWN_SECONDS:
                logger.warning(f"[BUTTON] COOLDOWN: Ignoring click (within {COOLDOWN_SECONDS}s)")
                return  # Ignore rapid clicks

            logger.info(f"[BUTTON] Setting starting_review=True, last_start_click={now}")
            st.session_state.last_start_click = now
            st.session_state.starting_review = True
            st.session_state.button_clicked = True  # Flag to trigger processing
            st.session_state.submission_content = st.session_state.get("submission_input", "")
            st.session_state.model_answer_content = st.session_state.get("model_answer_input", "")
            st.session_state.original_code_content = st.session_state.get("original_code_input", "")

        # Only render button if can_start is True
        if can_start:
            btn_log = f"[RENDER] Rendering button: can_start={can_start}"
            print(btn_log)
            logger.info(btn_log)
            with button_placeholder.container():
                start_btn = st.button("Start Review", type="primary", on_click=on_start_review, key="start_review_btn")
        else:
            # Clear button placeholder when button should not be shown
            button_placeholder.empty()
            logger.info(f"[RENDER] Button not rendered (can_start={can_start})")

# --- Process button click WITHOUT rerun (to avoid interrupting LangGraph) ---
if st.session_state.get("button_clicked"):
    print("[BUTTON_CLICKED] Processing button click - setting ready_to_process")
    logger.info("[BUTTON_CLICKED] Processing button click - setting ready_to_process")
    del st.session_state.button_clicked
    st.session_state.ready_to_process = True
    # DO NOT call st.rerun() here - it interrupts LangGraph streaming
    # The ready_to_process flag will be checked in the logic section below

# --- Logic & Display ---
config = {"configurable": {"thread_id": st.session_state.thread_id}}
if "db_session_id" in st.session_state:
    config["configurable"]["db_session_id"] = st.session_state.db_session_id

# save_step_log is now imported from services.py

# 2. Start Review Logic
# Check for starting_review flag (set by on_click callback)
logic_check = f"[LOGIC] Checking start conditions: starting_review={st.session_state.get('starting_review')}, db_session_id={st.session_state.get('db_session_id')}"
print(logic_check)
logger.info(logic_check)

# Start review when ready_to_process is True (after button click)
if st.session_state.get("ready_to_process", False):
    print("[LOGIC] ready_to_process detected - checking LangGraph state...")
    logger.info("[LOGIC] ready_to_process detected - checking LangGraph state...")

    # Check if there's an existing session and its state
    if "db_session_id" in st.session_state:
        # Clear flag BEFORE checking state to prevent re-entry
        del st.session_state.ready_to_process
        # Session exists - check if LangGraph is running or paused
        try:
            temp_snapshot = graph_app.get_state(config)

            if temp_snapshot and temp_snapshot.next:
                # LangGraph is RUNNING - ignore this request and clear flag
                ignore_log = f"[LOGIC] IGNORING REQUEST - LangGraph is already running (next={temp_snapshot.next})"
                print(ignore_log)
                logger.warning(ignore_log)
                # Clear ready_to_process flag to prevent re-processing
                if "ready_to_process" in st.session_state:
                    del st.session_state.ready_to_process
                # Do nothing - the graph will continue running

            elif temp_snapshot and not temp_snapshot.next and temp_snapshot.values:
                # LangGraph is PAUSED - resume it
                resume_log = "[LOGIC] *** RESUMING PAUSED REVIEW ***"
                print(resume_log)
                logger.info(resume_log)

                # Resume the graph from checkpoint (handled by display logic below)
                # The display logic will detect this state and show resume UI

            else:
                # Unexpected state
                error_log = f"[LOGIC] UNEXPECTED STATE - snapshot exists but no values"
                print(error_log)
                logger.error(error_log)

        except Exception as e:
            error_log = f"[LOGIC] Error checking LangGraph state: {e}"
            print(error_log)
            logger.error(error_log)

    else:
        # No session exists - start new review
        # Check if already starting (prevent duplicate processing)
        if st.session_state.get("starting_review", False) and "db_session_id" in st.session_state:
            # Already processing - clear flag and ignore
            del st.session_state.ready_to_process
            ignore_log = "[LOGIC] IGNORING - Already processing (starting_review=True, db_session_id exists)"
            print(ignore_log)
            logger.warning(ignore_log)
        else:
            # Clear flag BEFORE starting to prevent re-entry
            del st.session_state.ready_to_process

            import time as wait_time
            wait_time.sleep(1)  # Wait for UI to stabilize

            start_log = "[LOGIC] *** STARTING NEW REVIEW PROCESS ***"
            print(start_log)
            logger.info(start_log)

            # Get values from session state (saved by callback)
            submission = st.session_state.get("submission_content", "")
            model_answer = st.session_state.get("model_answer_content", "")
            original_code = st.session_state.get("original_code_content", "")

            if not submission:
                logger.error("[LOGIC] No submission content!")
                st.session_state.starting_review = False
                st.error("Submission text is required")
                st.stop()

            # Create Session in DB
            logger.info(f"[LOGIC] Creating DB session with thread_id={st.session_state.thread_id}")
            db = next(get_db())
            try:
                # Determine reviewer identity based on tier
                tier_label = "enterprise" if prompts.get_prompt_version() == "enterprise" else "lite"
                reviewer_identity = f"critic-agent-{tier_label}"

                new_session = ReviewSession(
                    id=st.session_state.thread_id, # Use Thread ID (UUID) as DB Session ID
                    user_id="demo_user", # User requesting review
                    reviewer_id=reviewer_identity, # AI Agent Identity
                    task_type=task_type,
                    submission_content=submission,
                    model_answer=model_answer
                )
                db.add(new_session)
                db.commit()
                st.session_state.db_session_id = new_session.id
                logger.info(f"[LOGIC] DB session created: db_session_id={st.session_state.db_session_id}")
            except Exception as e:
                # Clear starting flag on error
                logger.error(f"[LOGIC] DB session creation failed: {e}")
                st.session_state.starting_review = False
                st.error(f"Failed to create session: {e}")
                st.stop()
            finally:
                db.close()

            logger.info("[LOGIC] Starting graph workflow...")
            with st.status("🚀 Initializing Review Workflow...", expanded=True) as status:
                try:
                    # Initial Input
                    initial_state = {
                        "submission": submission,
                        "original_code": original_code,
                        "task_type": task_type,
                        "model_answer": model_answer,
                        "messages": [],
                        "status": "started",
                        "use_cli_bypass": use_cli_bypass,
                        "review_mode": review_mode,
                        "disable_web_search": disable_web_search
                    }

                    # Run Graph
                    # Pass db_session_id through config for token logging
                    config["configurable"]["db_session_id"] = st.session_state.db_session_id

                    # Record streaming start time
                    import time as _stream_time
                    st.session_state.streaming_start_time = _stream_time.time()
                    st.session_state.last_stream_event_time = _stream_time.time()

                    # Placeholder for live timeline updates
                    live_timeline = st.empty()

                    for event in graph_app.stream(initial_state, config=config):
                        # Update last event time on each event
                        st.session_state.last_stream_event_time = _stream_time.time()

                        # Event handling and DB logging
                        for key, value in event.items():
                            # st.write(f"Completed step: **{key}**") # Removed redundant status

                            # Extract relevant output based on node key
                            output_content = None
                            if key == "lint":
                                output_content = value.get("lint_result")
                                save_step_log(st.session_state.db_session_id, "Lint Check", output_content)
                            elif key == "research":
                                output_content = value.get("research_result")
                                save_step_log(st.session_state.db_session_id, "Research", output_content)
                            elif key == "analyze":
                                output_content = value.get("analysis_result")
                                save_step_log(st.session_state.db_session_id, "Analysis", output_content)
                            elif key == "fact_check":
                                output_content = value.get("fact_check_result")
                                save_step_log(st.session_state.db_session_id, "Fact Check", output_content)
                            elif key == "draft":
                                output_content = value.get("draft_review")
                                save_step_log(st.session_state.db_session_id, "Draft Review", output_content)
                            elif key == "critique":
                                output_content = value.get("critique_comment")
                                save_step_log(st.session_state.db_session_id, "Critique", output_content)
                            elif key == "refine":
                                output_content = value.get("draft_review")
                                save_step_log(st.session_state.db_session_id, "Refined Review", output_content)
                            elif key == "consistency":
                                output_content = value.get("consistency_result")
                                save_step_log(st.session_state.db_session_id, "Consistency Check", output_content)
                            elif key == "evaluate":
                                output_content = value.get("evaluation_result")
                                save_step_log(st.session_state.db_session_id, "Evaluation", output_content)
                            elif key == "scoring":
                                output_content = {"score": value.get("score"), "messages": value.get("messages")}
                                save_step_log(st.session_state.db_session_id, "Scoring", output_content)

                            # Update live timeline
                            with live_timeline.container():
                                st.markdown("### 🔄 Current Progress")
                                current_logs = get_session_state_from_db(st.session_state.db_session_id).get("logs", [])
                                render_timeline(current_logs)

                    # Check if workflow is completed
                    final_snapshot = graph_app.get_state(config)
                    if final_snapshot.values and not final_snapshot.next:
                        # Review completed - clear streaming timestamps
                        if "streaming_start_time" in st.session_state:
                            del st.session_state.streaming_start_time
                        if "last_stream_event_time" in st.session_state:
                            del st.session_state.last_stream_event_time
                        status.update(label="Review Completed", state="complete")
                    else:
                        # Workflow paused (waiting for approval) - keep timestamps for zombie detection
                        status.update(label="Workflow Paused (Waiting for Approval)", state="running")

                        # CRITICAL FIX: Clear timestamps so next resume doesn't think it's a zombie/running
                        if "streaming_start_time" in st.session_state:
                            del st.session_state.streaming_start_time
                        if "last_stream_event_time" in st.session_state:
                            del st.session_state.last_stream_event_time
                        logger.info("[LOGIC] Workflow paused - Cleared streaming timestamps to allow clean resume.")
                except Exception as e:
                    # Clear timestamps on error
                    if "streaming_start_time" in st.session_state:
                        del st.session_state.streaming_start_time
                    if "last_stream_event_time" in st.session_state:
                        del st.session_state.last_stream_event_time
                    st.error(f"An error occurred: {e}")
                    raise
                finally:
                    # Clear starting flag
                    if "starting_review" in st.session_state:
                        del st.session_state.starting_review

        # DO NOT call st.rerun() here - it interrupts LangGraph streaming
        # LangGraph will automatically trigger reruns as it streams

# --- Resume Review Logic (after button click) ---
if st.session_state.get("ready_to_resume", False):
    print("[RESUME] ready_to_resume detected - checking LangGraph state...")
    logger.info("[RESUME] ready_to_resume detected - checking LangGraph state...")

    # Clear flag immediately to prevent re-entry
    del st.session_state.ready_to_resume

    # Check if LangGraph is already running
    if "db_session_id" in st.session_state:
        try:
            resume_config = {"configurable": {"thread_id": st.session_state.thread_id}}
            resume_config["configurable"]["db_session_id"] = st.session_state.db_session_id
            temp_snapshot = graph_app.get_state(resume_config)

            if temp_snapshot and temp_snapshot.next:
                # --- ZOMBIE CHECK (STRICT) ---
                import time as _z_time
                ZOMBIE_TIMEOUT_RESUME = 180
                is_zombie_process = False

                last_evt_time = st.session_state.get("last_stream_event_time", 0)
                stream_start_time = st.session_state.get("streaming_start_time", 0)
                cur_time = _z_time.time()

                # Condition 1: No timestamps recorded
                if last_evt_time == 0:
                    is_zombie_process = True
                # Condition 2: Timeout exceeded
                elif (cur_time - last_evt_time) > ZOMBIE_TIMEOUT_RESUME:
                    is_zombie_process = True
                # Condition 3: Start time missing or inconsistencies
                elif stream_start_time == 0:
                    is_zombie_process = True
                # Condition 4: Last event is OLDER than start time (should be impossible if active)
                elif last_evt_time < stream_start_time:
                    is_zombie_process = True

                if not is_zombie_process:
                    # LangGraph is RUNNING (and active) - ignore this request
                    # Calculate remaining time for debug
                    elapsed = cur_time - last_evt_time
                    ignore_log = f"[RESUME] IGNORING REQUEST - LangGraph is ACTIVE (next={temp_snapshot.next}, elapsed={elapsed:.1f}s)"
                    print(ignore_log)
                    logger.warning(ignore_log)

                    if "resuming_review" in st.session_state:
                        del st.session_state.resuming_review
                else:
                    # Zombie or Resume-ready state detected
                    logger.warning(f"[RESUME] Zombie/Inactive process detected (next={temp_snapshot.next}). Forcing resume.")
                    resume_action = st.session_state.get("resume_action", "draft")
                    st.session_state.should_resume_now = True
            else:
                # LangGraph is PAUSED - proceed with resume
                resume_action = st.session_state.get("resume_action", "draft")
                logger.info(f"[RESUME] *** RESUMING REVIEW (action={resume_action}) ***")

                # Resume logic will be handled in the display section below
                # We just set the flag here to indicate we're ready to resume
                st.session_state.should_resume_now = True

        except Exception as e:
            error_log = f"[RESUME] Error checking LangGraph state: {e}"
            print(error_log)
            logger.error(error_log)
            if "resuming_review" in st.session_state:
                del st.session_state.resuming_review

# 3. Display Review Process (From DB or History)

# Check if viewing a history session
if "selected_history_id" in st.session_state:
    # History View Mode
    history_session_id = st.session_state.selected_history_id

    st.info(f"📖 履歴表示モード: セッション#{history_session_id}")

    db = next(get_db())

    # Load the session
    history_session = db.query(ReviewSession).filter(ReviewSession.id == history_session_id).first()

    if history_session:
        # Display submission
        st.divider()
        st.subheader("📝 提出物")
        with st.expander("Show Submission", expanded=False):
            st.text_area("Submission", value=history_session.submission_content, height=200, disabled=True)
            if history_session.model_answer:
                st.text_area("Model Answer", value=history_session.model_answer, height=150, disabled=True)

        # Display process timeline
        st.divider()
        st.subheader("🔍 Review Process Timeline")

        logs = db.query(ReviewStepLog).filter(ReviewStepLog.session_id == history_session_id).order_by(ReviewStepLog.created_at).all()

        for log in logs:
            with st.expander(f"{log.step_name} ({log.created_at.strftime('%H:%M:%S')})", expanded=False):
                try:
                    data = json.loads(log.output_data)
                    if isinstance(data, dict) and "content" in data:
                        content = data["content"]

                        # If json key exists, content is usually text (for Draft Review, etc.)
                        # and json contains the structured data
                        if "json" in data and isinstance(data["json"], dict):
                            # Display main content as text (for Draft Review, Refined Review, etc.)
                            content_text = extract_displayable_content(content)
                            st.markdown(content_text)

                            # Display structured JSON data separately
                            st.divider()
                            st.caption("📋 Structured JSON Data")
                            st.json(data["json"])
                        else:
                            # No json key - check if content itself is JSON (for Analyze, Fact Check, etc.)
                            is_json_content = False
                            json_data = None

                            # Try to parse as JSON
                            if isinstance(content, str):
                                try:
                                    json_data = json.loads(content)
                                    # Only treat as JSON if it's actually a JSON object/array
                                    # and not just a string that happens to be valid JSON
                                    if isinstance(json_data, (dict, list)):
                                        is_json_content = True
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            elif isinstance(content, dict):
                                json_data = content
                                is_json_content = True

                            # Display JSON with proper formatting
                            if is_json_content:
                                st.json(json_data)
                            else:
                                # Main Content - Use helper function to properly decode
                                content_text = extract_displayable_content(content)
                                st.markdown(content_text)

                        # Metadata Section
                        st.divider()
                        st.caption("📊 Process Metadata")

                        # Token Usage
                        if "token_usage" in data:
                            usage = data["token_usage"]
                            st.markdown(f"**Token Usage**: Input: `{usage.get('input')}` / Output: `{usage.get('output')}` / Total: `{usage.get('total')}`")

                        # Research Specifics
                        if "queries" in data:
                            st.markdown("**🔍 Generated Queries:**")
                            for q in data["queries"]:
                                st.code(q, language="text")

                        if "raw_result" in data:
                            with st.expander("📄 Raw Search Results"):
                                st.text(data["raw_result"])

                    else:
                        # Fallback for legacy strings or simple JSON
                        content_text = extract_displayable_content(log.output_data)
                        st.markdown(content_text)
                except json.JSONDecodeError:
                    st.markdown(log.output_data)

        # Display final result
        st.divider()
        final_result = db.query(ReviewResult).filter(ReviewResult.session_id == history_session_id).first()

        if final_result:
            st.success("✅ Review Completed")

            # Display Score if available
            if final_result.score is not None:
                score = final_result.score
                score_color, score_emoji = get_score_display_info(score)

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                            padding: 1.5rem; border-radius: 10px; border: 2px solid #0f3460;
                            text-align: center; margin-bottom: 1rem;">
                    <h2 style="margin: 0; color: #e94560;">
                        {score_emoji} {score_color} {score}/100
                    </h2>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🏆 Final Review")
            try:
                final_content_data = json.loads(final_result.final_comment)
                final_content = extract_displayable_content(final_content_data)
            except:
                final_content = final_result.final_comment
            st.markdown(final_content)

            # Display evaluation if available
            if final_result.evaluation_result:
                st.markdown("### ⚖️ Evaluation Result")
                try:
                    eval_data = json.loads(final_result.evaluation_result)
                    st.json(eval_data)
                except:
                    st.text(final_result.evaluation_result)

            if final_result.consistency_result:
                st.markdown("### 🔄 Consistency Check")
                try:
                    consist_data = json.loads(final_result.consistency_result)
                    st.json(consist_data)
                except:
                    st.text(final_result.consistency_result)
        else:
            st.warning("This session is not yet complete")
    else:
        st.error(f"Session #{history_session_id} not found")

    db.close()

elif "db_session_id" in st.session_state:
    # Get session state from DB
    db_state = get_session_state_from_db(st.session_state.db_session_id)
    current_session = db_state.get("session")
    is_resumed = current_session and current_session.status == "in_progress"
    logs = db_state.get("logs", [])

    # Extract state information
    next_step_from_db = db_state.get("next_step")
    has_draft = db_state.get("has_draft", False)
    has_analysis = db_state.get("has_analysis", False)
    draft_review_data = db_state.get("draft_review_data")

    # Restore session state from DB if needed
    if current_session and not st.session_state.get("submission_content"):
        st.session_state.submission_content = current_session.submission_content
        st.session_state.model_answer_content = current_session.model_answer or ""

    if is_resumed:
        st.info("🔄 **Resuming**: State restored from database history. Continuing execution.")

    st.divider()
    st.subheader("🔍 Review Process Timeline")
    # Placeholder for live timeline updates
    timeline_placeholder = st.container()
    with timeline_placeholder:
        render_timeline(logs)



    # 4. Current State & Actions
    # Try to get state from LangGraph checkpoint first
    snapshot = graph_app.get_state(config)

    # If checkpoint exists, use it (preferred)
    reconstructed_state = None
    if snapshot.next:
        state_values = snapshot.values
        use_checkpoint = True
        # Update disable_web_search from UI setting (user may have changed it)
        if "disable_web_search" in state_values:
            state_values["disable_web_search"] = disable_web_search
            graph_app.update_state(config, {"disable_web_search": disable_web_search})
    else:
        # Fallback: Reconstruct state from database
        reconstructed_state = reconstruct_langgraph_state_from_db(st.session_state.db_session_id)
        if reconstructed_state:
            # Apply UI settings (disable_web_search, review_mode, etc.)
            reconstructed_state["disable_web_search"] = disable_web_search
            reconstructed_state["review_mode"] = review_mode
            reconstructed_state["use_cli_bypass"] = use_cli_bypass
            # Restore state to LangGraph checkpoint
            graph_app.update_state(config, reconstructed_state)
            state_values = reconstructed_state
            use_checkpoint = False
            # Update snapshot after state restoration
            snapshot = graph_app.get_state(config)
        else:
            # Minimal fallback if reconstruction fails
            state_values = {}
            if draft_review_data:
                state_values["draft_review"] = draft_review_data
            if has_analysis:
                state_values["status"] = "analyzed"
            use_checkpoint = False

    # Execute resume logic if should_resume_now flag is set
    if st.session_state.get("should_resume_now", False):
        del st.session_state.should_resume_now
        resume_action = st.session_state.get("resume_action", "draft")

        logger.info(f"[RESUME] Executing resume action: {resume_action}")

        # Import time module once for both resume blocks
        import time as _resume_stream_time

        if resume_action == "draft":
            # Resume to draft generation
            with st.spinner("Generating Draft..."):
                if not snapshot.next:
                    # No checkpoint, ensure state is set
                    if reconstructed_state:
                        # Apply UI settings
                        reconstructed_state["disable_web_search"] = disable_web_search
                        reconstructed_state["review_mode"] = "standard"
                        reconstructed_state["use_cli_bypass"] = False
                        # Update state with as_node="fact_check" to simulate completing fact_check node
                        graph_app.update_state(config, reconstructed_state, as_node="fact_check")

                        verify_snapshot = graph_app.get_state(config)
                        logger.info(f"[Resume Debug] After update_state(fact_check): next={verify_snapshot.next}")

                # Get final snapshot state to verify
                final_snapshot = graph_app.get_state(config)
                logger.info(f"[Resume Debug] Before stream: next={final_snapshot.next}")

                if not final_snapshot.next:
                    st.error("⚠️ Failed to set LangGraph checkpoint.")
                    st.info("💡 Please start a new session.")
                    if "resuming_review" in st.session_state:
                        del st.session_state.resuming_review
                    st.stop()

                # Record streaming start time for resume
                st.session_state.streaming_start_time = _resume_stream_time.time()
                st.session_state.last_stream_event_time = _resume_stream_time.time()

                # FORCE state update to wake up the graph
                logger.info(f"[RESUME] Forcing state update to trigger execution. Config: {config}")
                graph_app.update_state(config, {"human_feedback": None})

                # Resume execution
                logger.info(f"[RESUME] Calling graph_app.stream(None)...")
                event_count = 0
                # Update timestamp before stream() to mark execution start
                st.session_state.last_stream_event_time = _resume_stream_time.time()
                for event in graph_app.stream(None, config=config):
                    event_count += 1
                    logger.info(f"[RESUME] Stream event received: {list(event.keys())}")
                    # Update last event time on each event
                    st.session_state.last_stream_event_time = _resume_stream_time.time()

                    for key, value in event.items():
                        logger.info(f"[Resume Debug] Processing node: {key}")
                        output_content = None
                        if key == "draft":
                            output_content = value.get("draft_review")
                            save_step_log(st.session_state.db_session_id, "Draft Review", output_content)
                        elif key == "critique":
                            output_content = value.get("critique_comment")
                            save_step_log(st.session_state.db_session_id, "Critique", output_content)
                        elif key == "refine":
                            output_content = value.get("draft_review")
                            save_step_log(st.session_state.db_session_id, "Refined Review", output_content)

                # Clear resuming flag and action
                if "resuming_review" in st.session_state:
                    del st.session_state.resuming_review
                if "resume_action" in st.session_state:
                    del st.session_state.resume_action

        elif resume_action == "finalize":
            # Resume to finalize review
            current_snapshot = graph_app.get_state(config)
            with st.spinner("Finalizing Review..."):
                if not current_snapshot.next:
                    # No checkpoint, ensure state is fully set
                    if not reconstructed_state:
                        reconstructed_state = reconstruct_langgraph_state_from_db(st.session_state.db_session_id)
                    if reconstructed_state:
                        # Apply UI settings
                        reconstructed_state["disable_web_search"] = disable_web_search
                        reconstructed_state["review_mode"] = "standard"
                        reconstructed_state["use_cli_bypass"] = False
                        reconstructed_state["human_feedback"] = None

                        # Update state with as_node="refine" to simulate completing refine node
                        graph_app.update_state(config, reconstructed_state, as_node="refine")

                        verify_snapshot = graph_app.get_state(config)
                        logger.info(f"[Resume Debug] After update_state: next={verify_snapshot.next}")
                else:
                    # Checkpoint exists, just clear human_feedback
                    graph_app.update_state(config, {"human_feedback": None})

                # Get final snapshot state to verify
                final_snapshot = graph_app.get_state(config)
                logger.info(f"[Resume Debug] Before stream: next={final_snapshot.next}")

                if not final_snapshot.next:
                    st.error("⚠️ Failed to set LangGraph checkpoint.")
                    st.info("💡 Please start a new session.")
                    if "resuming_review" in st.session_state:
                        del st.session_state.resuming_review
                    st.stop()

                # Record streaming start time for resume
                st.session_state.streaming_start_time = _resume_stream_time.time()
                st.session_state.last_stream_event_time = _resume_stream_time.time()

                # FORCE state update to wake up the graph (clearing feedback is good anyway)
                graph_app.update_state(config, {"human_feedback": None})

                # Resume from checkpoint
                logger.info(f"[RESUME] Calling graph_app.stream(None)...")
                event_count = 0
                # Update timestamp before stream() to mark execution start
                st.session_state.last_stream_event_time = _resume_stream_time.time()
                # Placeholder for live timeline updates
                live_timeline_resume = st.empty()

                for event in graph_app.stream(None, config=config):
                    event_count += 1
                    logger.info(f"[RESUME] Stream event received: {list(event.keys())}")
                    # Update last event time on each event
                    st.session_state.last_stream_event_time = _resume_stream_time.time()

                    for key, value in event.items():
                        logger.info(f"[Resume Debug] Processing node: {key}")
                        output_content = None
                        if key == "consistency":
                            output_content = value.get("consistency_result")
                            save_step_log(st.session_state.db_session_id, "Consistency Check", output_content)
                        elif key == "evaluate":
                            output_content = value.get("evaluation_result")
                            save_step_log(st.session_state.db_session_id, "Evaluation", output_content)
                        elif key == "scoring":
                            output_content = {"score": value.get("score"), "messages": value.get("messages")}
                            save_step_log(st.session_state.db_session_id, "Scoring", output_content)

                        # Update live timeline
                        with live_timeline_resume.container():
                            st.markdown("### 🔄 Current Progress")
                            current_logs = get_session_state_from_db(st.session_state.db_session_id).get("logs", [])
                            render_timeline(current_logs)

                # Clear resuming flag and action
                if "resuming_review" in st.session_state:
                    del st.session_state.resuming_review
                if "resume_action" in st.session_state:
                    del st.session_state.resume_action

    # Show action UI if session is in progress and we can determine next step
    should_show_actions = False
    if current_session and current_session.status == "in_progress":
        # Show actions if we have checkpoint or can determine state from DB
        if snapshot.next:
            should_show_actions = True
        elif next_step_from_db or draft_review_data or has_analysis:
            should_show_actions = True

    if should_show_actions:
        st.divider()
        st.subheader("⚡️ Actions")

        # Display LangGraph state clearly with zombie detection
        import time as _check_time
        ZOMBIE_TIMEOUT_SECONDS = 30  # Strict timeout

        is_zombie = False
        if snapshot.next:
            # Check if streaming is actually active (not zombie)
            last_event_time = st.session_state.get("last_stream_event_time", 0)
            stream_start_time = st.session_state.get("streaming_start_time", 0)

            if last_event_time > 0:
                time_since_last_event = _check_time.time() - last_event_time

                # Check 1: Timeouts
                if time_since_last_event > ZOMBIE_TIMEOUT_SECONDS:
                    is_zombie = True
                    zombie_log = f"[UI] ZOMBIE DETECTED: Timeout ({time_since_last_event:.1f}s > {ZOMBIE_TIMEOUT_SECONDS}s)"
                    print(zombie_log)
                    logger.warning(zombie_log)
                # Check 2: Start time consistency
                elif stream_start_time == 0:
                    is_zombie = True
                    zombie_log = f"[UI] ZOMBIE DETECTED: Missing streaming_start_time"
                    print(zombie_log)
                    logger.warning(zombie_log)
                elif last_event_time < stream_start_time:
                    is_zombie = True
                    zombie_log = f"[UI] ZOMBIE DETECTED: Invalid timestamp order (Last < Start)"
                    print(zombie_log)
                    logger.warning(zombie_log)
            else:
                # No timestamp recorded - likely zombie from start
                is_zombie = True
                zombie_log = f"[UI] ZOMBIE DETECTED: No streaming timestamp recorded"
                print(zombie_log)
                logger.warning(zombie_log)

        langgraph_status = "🔄 **実行中**" if (snapshot.next and not is_zombie) else "⏸️ **停止中**"
        next_nodes = str(snapshot.next) if snapshot.next else "None"
        status_log = f"[UI] LangGraph Status: {langgraph_status}, next={next_nodes}, is_zombie={is_zombie}"
        print(status_log)
        logger.info(status_log)

        if snapshot.next and not is_zombie:
            st.info(f"🔄 **LangGraph Running**: Executing next node `{next_nodes}`. Please wait for completion.")
        elif is_zombie:
            st.warning(f"⚠️ **Zombie State Detected**: No events received for {ZOMBIE_TIMEOUT_SECONDS}s. Use Resume button to continue.")
        else:
            st.info(f"⏸️ **LangGraph Paused**: Use Resume button to continue processing.")

        if is_resumed:
            if use_checkpoint:
                st.info("🔄 **Resuming**: State restored from checkpoint. Continuing execution.")
            else:
                st.info("🔄 **Resuming**: State restored from database history. Continuing execution.")

        col_a, col_b = st.columns([1, 2])

        with col_a:
            # Resume button configuration (shared by both resume buttons)
            import time as _resume_time
            RESUME_COOLDOWN_SECONDS = 2

            if state_values.get("status") == "analyzed":
                st.info("Analysis Complete. Proceed to Draft?")

                # Resume button with debounce protection
                def on_resume_draft():
                    # Check if already resuming
                    if st.session_state.get("resuming_review"):
                        logger.warning("[RESUME] ABORT: Already resuming")
                        return

                    # Cooldown check
                    last_resume = st.session_state.get("last_resume_click", 0)
                    now = _resume_time.time()

                    if now - last_resume < RESUME_COOLDOWN_SECONDS:
                        logger.warning(f"[RESUME] COOLDOWN: Ignoring click (within {RESUME_COOLDOWN_SECONDS}s)")
                        return

                    logger.info("[RESUME] Setting resuming_review=True")
                    st.session_state.last_resume_click = now
                    st.session_state.resuming_review = True
                    st.session_state.ready_to_resume = True
                    st.session_state.resume_action = "draft"  # Action type

                # Show button (will be disabled if LangGraph is actually running, not zombie)
                if st.session_state.get("resuming_review", False):
                    st.info("🔄 再開処理中...")
                else:
                    resume_draft_btn = st.button("Approve Analysis & Generate Draft", on_click=on_resume_draft, key="resume_draft_btn")

            elif state_values.get("draft_review"):
                st.success("Draft Ready for Final Review")

                # Resume button with debounce protection
                def on_resume_finalize():
                    # Check if already resuming
                    if st.session_state.get("resuming_review"):
                        logger.warning("[RESUME] ABORT: Already resuming")
                        return

                    # Cooldown check
                    last_resume = st.session_state.get("last_resume_click", 0)
                    now = _resume_time.time()

                    if now - last_resume < RESUME_COOLDOWN_SECONDS:
                        logger.warning(f"[RESUME] COOLDOWN: Ignoring click (within {RESUME_COOLDOWN_SECONDS}s)")
                        return

                    # --- ZOMBIE CHECK ---
                    # Get latest state to avoid stale closure capture
                    cur_snap = graph_app.get_state(config)
                    if cur_snap and cur_snap.next:
                        # Check for zombie
                        last_evt = st.session_state.get("last_stream_event_time", 0)
                        stream_start = st.session_state.get("streaming_start_time", 0)

                        is_zombie_cb = False
                        if last_evt == 0:
                            is_zombie_cb = True
                        elif (now - last_evt > ZOMBIE_TIMEOUT_SECONDS):
                            is_zombie_cb = True
                        elif stream_start == 0: # Missing start time
                            is_zombie_cb = True
                        elif last_evt < stream_start: # Invalid order
                            is_zombie_cb = True

                        if not is_zombie_cb:
                            logger.warning(f"[RESUME] ABORT: Active execution detected (next={cur_snap.next})")
                            st.warning(f"⚠️ Processing is currently active. Please wait for completion.")
                            return
                        else:
                            logger.info("[RESUME] Zombie detected in callback - Proceeding with resume")

                    logger.info("[RESUME] Setting resuming_review=True")
                    st.session_state.last_resume_click = now
                    st.session_state.resuming_review = True
                    st.session_state.ready_to_resume = True
                    st.session_state.resume_action = "finalize"  # Action type

                # Show button (will be disabled if LangGraph is actually running, not zombie)
                if st.session_state.get("resuming_review", False):
                    st.info("🔄 再開処理中...")
                else:
                    resume_finalize_btn = st.button("Approve & Finalize Review", on_click=on_resume_finalize, key="resume_finalize_btn")

        with col_b:
            if state_values.get("draft_review"):
                feedback = st.chat_input("Give feedback to refine the review...")
                if feedback:
                    # Update state with feedback
                    graph_app.update_state(config, {"human_feedback": feedback})
                    with st.spinner("Refining Review..."):
                        # Resume from checkpoint - LangGraph will continue from checkpoint
                        import time as _refine_time
                        st.session_state.last_stream_event_time = _refine_time.time()
                        # Placeholder for live timeline updates
                        live_timeline_refine = st.empty()

                        for event in graph_app.stream(None, config=config):
                            for key, value in event.items():
                                output_content = None
                                if key == "refine":
                                    output_content = value.get("draft_review")
                                    save_step_log(st.session_state.db_session_id, "Refined Review", output_content)

                                # Update live timeline
                                with live_timeline_refine.container():
                                    st.markdown("### 🔄 Current Progress")
                                    current_logs = get_session_state_from_db(st.session_state.db_session_id).get("logs", [])
                                    render_timeline(current_logs)
                    st.rerun()

                # DSPy Experiment
                with st.expander("🧪 DSPy Experiment", expanded=False):
                    if st.button("Generate DSPy Review"):
                        with st.spinner("Running DSPy..."):
                            dspy_result = generate_review_dspy(state_values["submission"], state_values["task_type"])
                            st.session_state.dspy_result = dspy_result
                            st.rerun()

                    if "dspy_result" in st.session_state:
                        st.markdown("#### DSPy Result")
                        st.markdown(st.session_state.dspy_result)

    # Check if session is completed
    if snapshot.values and not snapshot.next:
        # Completed
        st.success("✅ Review Session Completed")

        # Retrieve final results
        final_review_data = snapshot.values.get("refine_review") or snapshot.values.get("draft_review")
        evaluation_result = snapshot.values.get("evaluation_result")
        consistency_result = snapshot.values.get("consistency_result")
        score = snapshot.values.get("score")

        # Display Score if available
        if score is not None:
            st.markdown("### 📊 Score")
            score_color, score_emoji = get_score_display_info(score)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        padding: 1.5rem; border-radius: 10px; border: 2px solid #0f3460;
                        text-align: center; margin-bottom: 1rem;">
                <h2 style="margin: 0; color: #e94560;">
                    {score_emoji} {score_color} {score}/100
                </h2>
            </div>
            """, unsafe_allow_html=True)

        # Prepare for display
        final_content = ""
        if isinstance(final_review_data, dict):
            final_content = final_review_data.get("content", "")
        else:
            final_content = str(final_review_data) if final_review_data else ""

        st.markdown("### 🏆 Final Review")
        st.markdown(extract_displayable_content(final_review_data))

        # Display Evaluation Results if available
        if evaluation_result:
            st.markdown("### ⚖️ Evaluation Result")
            st.json(evaluation_result)

        if consistency_result:
            st.markdown("### 🔄 Consistency Check")
            st.json(consistency_result)

        # Save final result if not saved
        if "result_saved" not in st.session_state:
            db = next(get_db())

            # Serialize to JSON for DB storage (to_json_str is imported from services)
            res = ReviewResult(
                session_id=st.session_state.db_session_id,
                final_comment=to_json_str(final_review_data),
                score=score,
                evaluation_result=to_json_str(evaluation_result),
                consistency_result=to_json_str(consistency_result),
                is_manual_override=False
            )
            db.add(res)
            db.commit()
            db.close()
            st.session_state.result_saved = True

