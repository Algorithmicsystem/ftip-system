-- Migration 106: Ensure axiom_scores_daily has all columns required by the INSERT in persistence.py.
-- Uses ADD COLUMN IF NOT EXISTS so this is safe to re-run on any environment.
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS deployable_alpha_utility numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS regime_label text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS payload jsonb;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS outcome_payload jsonb;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS build_meta jsonb;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS signal_version text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS feature_version text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS snapshot_version text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS snapshot_id text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS evidence_backed_deployability_tier text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS trade_family text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS deployability_tier text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS size_band text;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS gross_opportunity numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS friction_burden numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS validated_edge numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS overall_coverage numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS overall_confidence numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS ml_adjusted_dau numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS moat_score numeric;
ALTER TABLE axiom_scores_daily ADD COLUMN IF NOT EXISTS intelligence_quality_score numeric;
