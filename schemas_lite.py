"""
JSON Output Schemas for Review System Nodes (Lite / OSS Version)

Note: This is the Lite version of the schemas for the OSS edition.
Enterprise features (such as hallucination propagation tracking, deep security taxonomies,
and agent negotiation protocols) are available in the Enterprise version.
"""

from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# 共通型
# =============================================================================

class Location(BaseModel):
    """提出物内の位置を示す"""
    section: str = Field(..., description="セクション名（例: 'プロンプト1 > # 役割'）")
    line_hint: Optional[str] = Field(None, description="行の特定に役立つヒント")
    quote: str = Field(..., description="実際のテキスト引用")
    note: Optional[str] = Field(None, description="補足説明")


class Evidence(BaseModel):
    """指摘の根拠"""
    section: str = Field(..., description="セクション名")
    quote: str = Field(..., description="実際のテキスト引用")
    note: Optional[str] = Field(None, description="補足説明")


# =============================================================================
# Router出力
# =============================================================================

class RouterOutput(BaseModel):
    """Routerノードの出力"""
    task_type: Literal["Prompt Engineering", "Programming"] = Field(
        ..., description="タスクの種類"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="判定の確信度")
    reasoning: str = Field(..., description="判定理由")


# =============================================================================
# Lint出力
# =============================================================================

class LintIssue(BaseModel):
    """Lintで検出された問題"""
    # Note: Enterprise version includes deeper security taxonomies
    type: str = Field(..., description="問題の種類（例: syntax_error, security_risk 等）")
    severity: Literal["error", "warning", "info"] = Field(..., description="深刻度")
    location: Evidence = Field(..., description="問題の箇所（根拠必須）")
    description: str = Field(..., description="問題の説明")
    suggestion: str = Field(..., description="修正案")


class LintOutput(BaseModel):
    """Lintノードの出力"""
    status: Literal["pass", "issues_found"] = Field(..., description="ステータス")
    issues: List[LintIssue] = Field(default_factory=list, description="検出された問題")


# =============================================================================
# Research出力
# =============================================================================

class SearchQuery(BaseModel):
    """検索クエリ"""
    query: str = Field(..., description="検索クエリ")
    intent: str = Field(..., description="検索の意図")


class SearchResult(BaseModel):
    """検索結果"""
    source: str = Field(..., description="ソース名")
    url: Optional[str] = Field(None, description="URL")
    relevant_content: str = Field(..., description="関連する内容の抜粋")
    relevance: Literal["high", "medium", "low"] = Field(..., description="関連度")


class ResearchOutput(BaseModel):
    """Researchノードの出力"""
    queries: List[SearchQuery] = Field(default_factory=list, description="実行した検索クエリ")
    results: List[SearchResult] = Field(default_factory=list, description="検索結果")
    summary: Optional[str] = Field(None, description="検索結果の要約")


# =============================================================================
# Analyze出力
# =============================================================================

class HallucinationEvidence(BaseModel):
    """ハルシネーションの証拠"""
    created_at: Evidence = Field(..., description="創作された箇所")
    original_missing: Evidence = Field(..., description="元データで欠落している箇所")
    # Note: Enterprise version includes 'propagated_to' for tracking downstream impact


class FatalFlaw(BaseModel):
    """致命的欠陥"""
    # Note: Enterprise version uses a strict Enum for proprietary taxonomies
    type: str = Field(..., description="欠陥の種類（例: hallucination, security_risk 等）")
    severity: Literal["critical", "high"] = Field(default="critical", description="深刻度")
    description: str = Field(..., description="問題の説明")
    evidence: HallucinationEvidence = Field(..., description="根拠（必須）")
    business_impact: str = Field(..., description="実務での影響")


class Strength(BaseModel):
    """強み"""
    type: str = Field(..., description="強みの種類")
    description: str = Field(..., description="強みの説明")
    evidence: Evidence = Field(..., description="根拠（必須）")


class Improvement(BaseModel):
    """改善点"""
    type: str = Field(..., description="改善点の種類")
    description: str = Field(..., description="改善点の説明")
    evidence: Evidence = Field(..., description="根拠（必須）")
    suggestion: str = Field(..., description="改善案")
    example_code: Optional[str] = Field(None, description="修正後のコード例")


class AnalyzeOutput(BaseModel):
    """Analyzeノードの出力"""
    verdict: Literal["PASS", "FAIL"] = Field(..., description="判定")
    skill_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ..., description="推定スキルレベル"
    )
    needs_research: bool = Field(
        default=False,
        description="より詳細な分析のためにWeb検索が必要な場合はTrue"
    )
    research_queries: List[str] = Field(
        default_factory=list,
        description="詳細調査のための検索クエリ"
    )
    fatal_flaws: List[FatalFlaw] = Field(
        default_factory=list,
        description="致命的欠陥（1つでもあればFAIL）"
    )
    strengths: List[Strength] = Field(default_factory=list, description="強み")
    improvements: List[Improvement] = Field(default_factory=list, description="改善点")
    next_topics: List[str] = Field(default_factory=list, description="次に学ぶべきトピック")


# =============================================================================
# Fact Check出力
# =============================================================================

class VerificationResult(BaseModel):
    """検証結果"""
    claim_path: str = Field(..., description="検証した主張のパス")
    claim_summary: str = Field(..., description="主張の要約")
    evidence_location: str = Field(..., description="根拠の箇所")
    verified: bool = Field(..., description="検証結果")
    actual_content: str = Field(..., description="実際に見つかった内容")
    note: Optional[str] = Field(None, description="補足")


class MissedIssue(BaseModel):
    """Analyzeで見落とされた問題"""
    type: str = Field(..., description="問題の種類")
    description: str = Field(..., description="問題の説明")
    evidence: Evidence = Field(..., description="根拠")


class FalseClaim(BaseModel):
    """誤った主張"""
    claim_path: str = Field(..., description="誤った主張のパス")
    issue: str = Field(..., description="何が誤っているか")


class FactCheckOutput(BaseModel):
    """Fact Checkノードの出力"""
    verification_results: List[VerificationResult] = Field(
        default_factory=list, description="検証結果"
    )
    missed_issues: List[MissedIssue] = Field(
        default_factory=list, description="見落とされた問題"
    )
    false_claims: List[FalseClaim] = Field(
        default_factory=list, description="誤った主張"
    )
    overall_validity: bool = Field(..., description="全体的な妥当性")


# =============================================================================
# Critique出力
# =============================================================================

class SeverityMismatchCheck(BaseModel):
    """Severity Mismatchチェック"""
    has_fatal_flaws: bool = Field(..., description="致命的欠陥があるか")
    fatal_flaw_count: int = Field(..., description="致命的欠陥の数")
    json_verdict: Literal["PASS", "FAIL"] = Field(..., description="JSONの判定")
    draft_verdict: Literal["PASS", "FAIL"] = Field(..., description="Draftの判定")
    mismatch: bool = Field(..., description="不一致があるか")
    note: Optional[str] = Field(None, description="補足")


class MissingInDraft(BaseModel):
    """Draftに含まれていない項目"""
    json_path: str = Field(..., description="JSONのパス")
    content: str = Field(..., description="内容")
    issue: str = Field(..., description="問題点")


class JsonDraftConsistency(BaseModel):
    """JSON↔Draft整合性チェック"""
    verdict_match: bool = Field(..., description="判定が一致しているか")
    missing_in_draft: List[MissingInDraft] = Field(
        default_factory=list, description="Draftに含まれていない項目"
    )
    extra_in_draft: List[str] = Field(
        default_factory=list, description="JSONにない項目がDraftにある"
    )


class ToneIssue(BaseModel):
    """トーンの問題"""
    type: str = Field(..., description="問題の種類")
    severity: Literal["error", "warning"] = Field(..., description="深刻度")
    location: str = Field(..., description="問題の箇所")
    quote: str = Field(..., description="問題のある文")
    suggestion: str = Field(..., description="修正案")


class ToneChecklist(BaseModel):
    """トーンチェックリスト"""
    professional: bool = Field(..., description="プロフェッショナルか")
    respectful: bool = Field(..., description="敬意があるか")
    not_condescending: bool = Field(..., description="上から目線でないか")
    not_overly_familiar: bool = Field(..., description="馴れ馴れしくないか")
    concrete_examples: bool = Field(..., description="具体例があるか")
    constructive: bool = Field(..., description="建設的か")


class ToneAndManner(BaseModel):
    """トーン&マナーチェック"""
    overall_appropriate: bool = Field(..., description="全体的に適切か")
    issues: List[ToneIssue] = Field(default_factory=list, description="問題")
    checklist: ToneChecklist = Field(..., description="チェックリスト")


class Revision(BaseModel):
    """修正指示"""
    type: str = Field(..., description="修正の種類")
    priority: Literal["critical", "high", "medium", "low"] = Field(..., description="優先度")
    instruction: str = Field(..., description="修正指示")


class CritiqueOverall(BaseModel):
    """Critique総合判定"""
    approve: bool = Field(..., description="承認するか")
    revisions: List[Revision] = Field(
        default_factory=list, description="修正指示リスト"
    )
    # Note: Enterprise version includes 'accepted_persuasions' for agent negotiation logs


class CritiqueOutput(BaseModel):
    """Critiqueノードの出力"""
    severity_mismatch_check: SeverityMismatchCheck = Field(
        ..., description="Severity Mismatchチェック"
    )
    # Note: Enterprise version includes 'strictness_check' for detecting leniency
    json_draft_consistency: JsonDraftConsistency = Field(
        ..., description="JSON↔Draft整合性チェック"
    )
    tone_and_manner: ToneAndManner = Field(..., description="トーン&マナーチェック")
    overall: CritiqueOverall = Field(..., description="総合判定")
