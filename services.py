"""
ビジネスロジック層
UI制御以外のロジックを集約
"""
import json
from database import get_db, ReviewSession, ReviewResult, ReviewStepLog
from typing import Optional, Dict, Any, List, Tuple


def extract_displayable_content(data):
    """
    Extract and format content from various data structures for UI display.

    Handles:
    - String content
    - Dict with 'content' key
    - List of dicts with 'text' key (LangChain format)
    - JSON-encoded strings

    Args:
        data: The data to extract content from (can be str, dict, or list)

    Returns:
        str: Formatted content ready for display
    """
    if data is None:
        return ""

    # If it's a string, try to parse as JSON first
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # Not JSON, return as-is
            return data

    # If it's a dict with 'content' key
    if isinstance(data, dict):
        if "content" in data:
            content = data["content"]
            # Recursively process the content
            return extract_displayable_content(content)
        else:
            # Return JSON prettified
            return json.dumps(data, indent=2, ensure_ascii=False)

    # If it's a list (LangChain format)
    if isinstance(data, list):
        extracted_texts = []
        for item in data:
            if isinstance(item, dict):
                # LangChain format: [{'type': 'text', 'text': '...'}]
                if 'text' in item:
                    extracted_texts.append(item['text'])
                elif 'content' in item:
                    extracted_texts.append(str(item['content']))
                else:
                    extracted_texts.append(json.dumps(item, indent=2, ensure_ascii=False))
            else:
                extracted_texts.append(str(item))

        return "\n\n".join(extracted_texts)

    # Fallback: convert to string
    return str(data)


def save_step_log(session_id: str, step_name: str, output_data: Any):
    """
    Save step log to database.

    Args:
        session_id: Session ID
        step_name: Name of the step
        output_data: Output data to save (will be serialized to JSON)
    """
    from langchain_core.messages import BaseMessage

    db = next(get_db())
    try:
        # Helper function to recursively convert AIMessage/BaseMessage to dict
        def convert_messages_to_dict(obj):
            if isinstance(obj, BaseMessage):
                # Convert AIMessage/HumanMessage to simple dict
                return {
                    "type": obj.__class__.__name__,
                    "content": obj.content
                }
            elif isinstance(obj, dict):
                return {k: convert_messages_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_messages_to_dict(item) for item in obj]
            else:
                return obj

        # Convert AIMessage objects to serializable format
        serializable_data = convert_messages_to_dict(output_data)

        # Convert dict/list to string if needed
        if isinstance(serializable_data, (dict, list)):
            output_str = json.dumps(serializable_data, ensure_ascii=False, indent=2)
        else:
            output_str = str(serializable_data)

        log = ReviewStepLog(
            session_id=session_id,
            step_name=step_name,
            output_data=output_str,
            status="success"
        )
        db.add(log)
        db.commit()
    except Exception as e:
        raise Exception(f"DB Error: {e}")
    finally:
        db.close()


def run_database_migrations():
    """
    Run database migrations to add missing columns.
    Returns True if migrations were run, False otherwise.
    """
    try:
        import sqlite3
        conn = sqlite3.connect("reviews.db")
        cursor = conn.cursor()

        migrations_run = False

        # Migration 1: Add status column to review_sessions
        cursor.execute("PRAGMA table_info(review_sessions)")
        session_columns = [column[1] for column in cursor.fetchall()]
        if "status" not in session_columns:
            cursor.execute("ALTER TABLE review_sessions ADD COLUMN status VARCHAR DEFAULT 'in_progress'")
            # Update existing records
            cursor.execute("""
                UPDATE review_sessions
                SET status = 'completed'
                WHERE id IN (SELECT session_id FROM review_results)
            """)
            migrations_run = True

        # Migration 2: Add score column to review_results
        cursor.execute("PRAGMA table_info(review_results)")
        result_columns = [column[1] for column in cursor.fetchall()]
        if "score" not in result_columns:
            cursor.execute("ALTER TABLE review_results ADD COLUMN score INTEGER")
            migrations_run = True

        conn.commit()
        conn.close()
        return migrations_run
    except Exception as e:
        raise Exception(f"Database migration error: {e}")


def get_session_state_from_db(session_id: str) -> Dict[str, Any]:
    """
    Reconstruct session state from database logs.

    Args:
        session_id: Session ID

    Returns:
        dict: Reconstructed state with keys:
            - completed_steps: set of completed step names
            - last_step: last completed step name
            - next_step: next step name (if determinable)
            - has_draft: whether draft review exists
            - has_analysis: whether analysis exists
            - draft_review_data: draft review data if available
            - session: ReviewSession object
    """
    db = next(get_db())
    try:
        session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
        if not session:
            return {}

        logs = db.query(ReviewStepLog).filter(ReviewStepLog.session_id == session_id).order_by(ReviewStepLog.created_at).all()

        # Determine current state from DB logs
        completed_steps = {log.step_name for log in logs}
        last_step = logs[-1].step_name if logs else None

        # Step flow mapping
        step_flow = {
            "Lint Check": "Research",
            "Research": "Analysis",
            "Analysis": "Fact Check",
            "Fact Check": "Draft Review",
            "Draft Review": "Critique",
            "Critique": "Refined Review",
            "Refined Review": "Consistency Check",
            "Consistency Check": "Evaluation",
            "Evaluation": "Scoring",
            "Scoring": None  # End
        }

        # Determine next step
        next_step = step_flow.get(last_step) if last_step else None
        has_draft = "Draft Review" in completed_steps or "Refined Review" in completed_steps
        has_analysis = "Analysis" in completed_steps

        # Extract draft_review from logs if available
        draft_review_data = None
        for log in reversed(logs):  # Get the latest
            if log.step_name in ["Draft Review", "Refined Review"]:
                try:
                    data = json.loads(log.output_data)
                    if isinstance(data, dict):
                        draft_review_data = data
                    break
                except:
                    pass

        return {
            "completed_steps": completed_steps,
            "last_step": last_step,
            "next_step": next_step,
            "has_draft": has_draft,
            "has_analysis": has_analysis,
            "draft_review_data": draft_review_data,
            "session": session,
            "logs": logs
        }
    finally:
        db.close()


def reconstruct_langgraph_state_from_db(session_id: str) -> Dict[str, Any]:
    """
    Reconstruct LangGraph ReviewState from database logs.
    This allows resuming a session even when checkpoint is lost (e.g., after server restart).

    Args:
        session_id: Session ID

    Returns:
        dict: Reconstructed ReviewState compatible with LangGraph
    """
    db = next(get_db())
    try:
        session = db.query(ReviewSession).filter(ReviewSession.id == session_id).first()
        if not session:
            return {}

        logs = db.query(ReviewStepLog).filter(ReviewStepLog.session_id == session_id).order_by(ReviewStepLog.created_at).all()

        # Map step names to state keys
        step_to_state_key = {
            "Lint Check": "lint_result",
            "Research": "research_result",
            "Analysis": "analysis_result",
            "Fact Check": "fact_check_result",
            "Draft Review": "draft_review",
            "Refined Review": "draft_review",  # Latest draft takes precedence
            "Critique": "critique_comment",
            "Consistency Check": "consistency_result",
            "Evaluation": "evaluation_result",
            "Scoring": "score"
        }

        # Initialize state with session data
        state = {
            "submission": session.submission_content,
            "task_type": session.task_type,
            "model_answer": session.model_answer or None,
            "original_code": None,  # Not stored in DB, would need to be added if needed
            "messages": [],  # Messages not stored in step logs
            "human_feedback": None,
            "status": "in_progress",
            "use_cli_bypass": False,  # Default, not stored in DB
            "review_mode": "standard",  # Default, not stored in DB
            "disable_web_search": False,  # Default, not stored in DB
            "refine_count": 0,
            "credits_charged": None,
            "pricing_info": None,
        }

        # Extract data from logs
        for log in logs:
            state_key = step_to_state_key.get(log.step_name)
            if state_key:
                try:
                    data = json.loads(log.output_data)
                    if state_key == "score":
                        # Score is stored as {"score": 85, ...}
                        if isinstance(data, dict) and "score" in data:
                            state["score"] = data["score"]
                    else:
                        state[state_key] = data
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, store as string
                    if state_key == "score":
                        try:
                            state["score"] = int(log.output_data)
                        except (ValueError, TypeError):
                            pass
                    else:
                        state[state_key] = log.output_data

        # Determine status based on last step
        if logs:
            last_step = logs[-1].step_name
            if last_step == "Scoring":
                state["status"] = "completed"
            elif last_step in ["Fact Check", "Refined Review"]:
                # These are interrupt points
                state["status"] = "analyzed" if last_step == "Fact Check" else "draft_ready"

        # Count refine iterations
        refine_count = sum(1 for log in logs if log.step_name == "Refined Review")
        state["refine_count"] = refine_count

        # Get score from ReviewResult if available
        result = db.query(ReviewResult).filter(ReviewResult.session_id == session_id).first()
        if result and result.score is not None:
            state["score"] = result.score

        return state
    finally:
        db.close()


def get_score_display_info(score: Optional[int]) -> Tuple[str, str]:
    """
    Get score display information (color and emoji).

    Args:
        score: Score value (0-100)

    Returns:
        tuple: (color_emoji, display_emoji)
    """
    if score is None:
        return "", ""

    if score >= 90:
        return "🟢", "🌟"
    elif score >= 80:
        return "🟡", "✨"
    elif score >= 70:
        return "🟠", "👍"
    else:
        return "🔴", "📝"


def get_status_badge(status: str) -> str:
    """
    Get status badge text.

    Args:
        status: Session status

    Returns:
        str: Status badge text
    """
    status_map = {
        "completed": "✅ 完了",
        "in_progress": "🔄 進行中",
        "failed": "❌ 失敗"
    }
    return status_map.get(status, status)


def to_json_str(data: Any) -> str:
    """
    Convert data to JSON string for database storage.

    Args:
        data: Data to convert

    Returns:
        str: JSON string (empty string if None)
    """
    if data is None:
        return ""
    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False)
    return str(data)

