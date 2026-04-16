"""
履歴ギャラリーページ
過去のレビューセッションを閲覧・管理できます。
"""
import streamlit as st
from database import get_db, ReviewSession, ReviewResult
from services import get_status_badge, get_score_display_info

st.set_page_config(page_title="履歴ギャラリー - Teacher's Copilot", layout="wide")

st.header("📚 履歴ギャラリー")
st.markdown("過去のレビューセッションを閲覧・管理できます。")

db = next(get_db())
# Get all review sessions, ordered by creation date (newest first)
all_sessions = db.query(ReviewSession).order_by(ReviewSession.created_at.desc()).all()

if not all_sessions:
    st.info("レビュー履歴がありません。新規レビューを作成してください。")
else:
    # Display sessions in a card layout
    for idx, session in enumerate(all_sessions):
        # Get result data
        result = db.query(ReviewResult).filter(ReviewResult.session_id == session.id).first()

        # Format date
        created_date = session.created_at.strftime("%Y年%m月%d日 %H:%M")

        # Get status badge
        status_badge = get_status_badge(session.status) if session.status else ""

        # Score display
        score_display = ""
        if result and result.score is not None:
            score = result.score
            score_color, _ = get_score_display_info(score)
            score_display = f"{score_color} **{score}/100**"

        # Submission preview
        submission_preview = session.submission_content[:100] + "..." if len(session.submission_content) > 100 else session.submission_content

        # Card container
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"### #{session.id[:8]}... - {created_date}")
                st.caption(f"**タスクタイプ**: {session.task_type} | {status_badge}")
                if score_display:
                    st.markdown(f"**スコア**: {score_display}")
                st.markdown(f"**提出物プレビュー**: {submission_preview}")

            with col2:
                if st.button("詳細を表示", key=f"view_{session.id}", use_container_width=True):
                    st.session_state.selected_history_id = session.id
                    # メインページに遷移（app.py）
                    st.switch_page("app.py")

            with col3:
                # Resume button for in_progress sessions
                if session.status == "in_progress":
                    if st.button("再開", key=f"resume_{session.id}", use_container_width=True, type="primary"):
                        st.session_state.thread_id = session.id
                        st.session_state.db_session_id = session.id
                        # メインページに遷移して再開
                        st.switch_page("app.py")

            st.divider()

db.close()

