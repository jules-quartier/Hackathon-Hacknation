import { useEffect, useState } from "react";
import { API_URL } from "./config";
import { Ring } from "./Ring";

type SessionMeta = { session_id: string; mode: string; has_summary?: boolean; started_at?: number };

type Summary = {
  session_id: string;
  operator_type: string;
  flight_duration_sec: number;
  inspection_coverage_percent: number;
  insulators_inspected: string[];
  min_cable_distance: number;
  safety_violations_count: number;
  max_speed_recorded: number;
  max_speed_near_structures?: number;
  hover_stability_score: number;
  defects_found: string[];
  defects_missed: string[];
  road_crossings?: Array<{ min_altitude: number; mean_speed: number; low: boolean }>;
  low_road_crossings?: number;
  predicted_conflicts?: number;
  guardian_interventions?: number;
  questions_asked?: number;
  answers_given?: number;
  notes_recorded?: number;
  knowledge_learned?: string[];
  ai_usage?: { calls: number; tokens: number; cost_usd: number };
  key_maneuvers?: string[];
  operational_summary?: string;
  coaching_points?: string[];
  mastery?: Mastery;
};

type MasteryItem = { slot: string; name: string; why: string[]; expert_rule: string; expert_words?: string | null };
type Mastery = {
  mastered: MasteryItem[];
  practice: MasteryItem[];
  not_practised: string[];
  quiz: { asked: number; right: number; partly: number; wrong: number };
};

type ComparisonReport = {
  expert_session_id: string;
  novice_session_id: string;
  overall_score: number;
  coverage_comparison: { expert_inspected_count: number; novice_inspected_count: number; missed_targets: string[]; commentary: string };
  safety_compliance: {
    min_cable_distance_expert: number;
    min_cable_distance_novice: number;
    violations_novice: number;
    guardian_interventions_novice?: number;
    compliance_rating: string;
    commentary: string;
  };
  technique_and_stability: { flight_duration_ratio: number; hover_discipline: string; knowledge_adherence: string };
  key_strengths: string[];
  areas_for_improvement: string[];
  instructor_verdict: string;
};

function Metric({ value, label, tone = "" }: { value: React.ReactNode; label: string; tone?: string }) {
  return (
    <div className={`metric ${tone}`}>
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

export function FlightComparison({ currentSessionId }: { currentSessionId: string | null }) {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [expertId, setExpertId] = useState("");
  const [noviceId, setNoviceId] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [report, setReport] = useState<ComparisonReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/sessions`)
      .then((r) => r.json())
      .then((data) => {
        const list: SessionMeta[] = data.sessions || [];
        setSessions(list);
        const experts = list.filter((s) => s.mode === "expert");
        const novices = list.filter((s) => s.mode === "novice" || s.mode === "tutor");
        if (experts.length && !expertId) setExpertId(experts[0].session_id);
        if (novices.length && !noviceId) setNoviceId(novices[0].session_id);
      })
      .catch((e) => console.error("Failed to load sessions:", e));
    if (!currentSessionId) return;
    setLoadingSummary(true);
    fetch(`${API_URL}/session/${currentSessionId}/summary`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => data && setSummary(data.summary))
      .catch(() => {})
      .finally(() => setLoadingSummary(false));
  }, [currentSessionId]);

  const compare = async () => {
    if (!expertId || !noviceId) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/session/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expert_session_id: expertId, novice_session_id: noviceId }),
      });
      const data = await res.json();
      if (data.comparison) setReport(data.comparison);
    } catch (e) {
      console.error("Comparison error:", e);
    } finally {
      setLoading(false);
    }
  };

  const s = summary;
  const label = (x: SessionMeta) =>
    `${x.session_id}${x.started_at ? " · " + new Date(x.started_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""}`;

  return (
    <div className="debrief">
      {loadingSummary && <div className="loading-bar" />}
      {s ? (
        <div className="panel">
          <div className="panel-head">
            <span className="eyebrow">Flight debrief</span>
            <span className={`chip ${s.operator_type === "expert" ? "ai" : "ok"}`}>{s.operator_type}</span>
          </div>
          <div className="metric-grid">
            <Metric value={`${s.flight_duration_sec}s`} label="Duration" />
            <Metric value={`${s.inspection_coverage_percent}%`} label="Coverage" tone={s.inspection_coverage_percent >= 99 ? "good" : ""} />
            <Metric value={`${s.min_cable_distance}m`} label="Min cable" tone={s.min_cable_distance < 2 ? "bad" : "good"} />
            <Metric value={s.safety_violations_count} label="Violations" tone={s.safety_violations_count ? "bad" : "good"} />
            <Metric value={`${s.hover_stability_score || "—"}`} label="Hover /10" />
            <Metric value={`${s.defects_found.length}/${s.defects_found.length + s.defects_missed.length}`} label="Defects found" tone={s.defects_missed.length ? "warn" : "good"} />
            <Metric value={s.predicted_conflicts ?? 0} label="Predicted conflicts" tone={s.predicted_conflicts ? "warn" : ""} />
            {s.operator_type === "expert" ? (
              <>
                <Metric value={s.knowledge_learned?.length ?? 0} label="Rules learned" tone="ai" />
                <Metric value={`$${(s.ai_usage?.cost_usd ?? 0).toFixed(3)}`} label={`AI · ${s.ai_usage?.calls ?? 0} calls`} tone="ai" />
              </>
            ) : (
              <>
                <Metric value={s.guardian_interventions ?? 0} label="Guardian saves" tone={s.guardian_interventions ? "warn" : "good"} />
                <Metric value={s.low_road_crossings ?? 0} label="Low road crossings" tone={s.low_road_crossings ? "warn" : "good"} />
              </>
            )}
          </div>
          {s.operational_summary && <p className="summary-text">{s.operational_summary}</p>}
          <div className="two-col">
            {s.key_maneuvers?.length ? (
              <div className="section">
                <h4>Key manoeuvres</h4>
                <ul className="bullets">{s.key_maneuvers.map((m) => <li key={m}>{m}</li>)}</ul>
              </div>
            ) : null}
            {s.coaching_points?.length ? (
              <div className="section">
                <h4>Next flight</h4>
                <ul className="bullets">{s.coaching_points.map((m) => <li key={m}>{m}</li>)}</ul>
              </div>
            ) : null}
          </div>
          {s.mastery && (s.mastery.mastered.length > 0 || s.mastery.practice.length > 0 || s.mastery.quiz.asked > 0) && (
            <div className="section mastery">
              {s.mastery.quiz.asked > 0 && (
                <p className="dim" style={{ marginTop: 8 }}>
                  Predicted the expert's decision: {s.mastery.quiz.right} right, {s.mastery.quiz.partly} almost, {s.mastery.quiz.wrong} wrong (
                  {s.mastery.quiz.asked} asked)
                </p>
              )}
              <div className="two-col">
                <div>
                  <h4>Mastered</h4>
                  {s.mastery.mastered.length ? (
                    s.mastery.mastered.map((m) => (
                      <div key={m.slot} className="mastery-item ok">
                        <b>✓ {m.name}</b>
                        <span>{m.why[0]}</span>
                      </div>
                    ))
                  ) : (
                    <p className="dim">Nothing yet.</p>
                  )}
                </div>
                <div>
                  <h4>Practice next</h4>
                  {s.mastery.practice.length ? (
                    s.mastery.practice.map((m) => (
                      <div key={m.slot} className="mastery-item warn">
                        <b>↻ {m.name}</b>
                        <span>{m.why[0]}</span>
                        {m.expert_words ? <em>The expert: “{m.expert_words}”</em> : <em>The expert: {m.expert_rule}</em>}
                      </div>
                    ))
                  ) : (
                    <p className="dim">Nothing to repeat.</p>
                  )}
                </div>
              </div>
              {s.mastery.not_practised.length > 0 && (
                <p className="dim" style={{ marginTop: 6 }}>
                  Not practised this flight: {s.mastery.not_practised.join(", ")}.
                </p>
              )}
            </div>
          )}
          {s.knowledge_learned?.length ? (
            <div className="section">
              <h4>Rules taught this flight</h4>
              <div className="target-chips">{s.knowledge_learned.map((k) => <span key={k} className="tchip kind-hypothesis">{k}</span>)}</div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="panel">
          <span className="eyebrow">Flight debrief</span>
          <p className="summary-text dim">End a flight to see its debrief: coverage, safety, predicted conflicts, Guardian saves, what the apprentice learned and what it cost.</p>
        </div>
      )}

      <div className="panel">
        <div className="panel-head">
          <span className="eyebrow">Expert vs novice</span>
        </div>
        <div className="compare-grid">
          <div>
            <label>Expert flight</label>
            <select className="select" value={expertId} onChange={(e) => setExpertId(e.target.value)}>
              <option value="">Select…</option>
              {sessions.filter((x) => x.mode === "expert").map((x) => <option key={x.session_id} value={x.session_id}>{label(x)}</option>)}
            </select>
          </div>
          <div>
            <label>Novice flight</label>
            <select className="select" value={noviceId} onChange={(e) => setNoviceId(e.target.value)}>
              <option value="">Select…</option>
              {sessions.filter((x) => x.mode === "novice" || x.mode === "tutor").map((x) => <option key={x.session_id} value={x.session_id}>{label(x)}</option>)}
            </select>
          </div>
        </div>
        <button className="btn btn-primary" style={{ width: "100%" }} onClick={compare} disabled={loading || !expertId || !noviceId}>
          {loading ? "Analysing…" : "Compare flights"}
        </button>
        {loading && <div className="loading-bar" style={{ marginTop: 8 }} />}
      </div>

      {report && (
        <div className="panel">
          <div className="score-hero">
            <Ring size={78} stroke={7} value={report.overall_score} total={100} secondary={report.overall_score >= 80 ? report.overall_score : 0}>
              <text x="39" y="45" textAnchor="middle" fill="#fff" fontFamily="Orbitron" fontSize="19" fontWeight="700">
                {Math.round(report.overall_score)}
              </text>
            </Ring>
            <div>
              <h3>Instructor evaluation</h3>
              <span className={`rating ${report.safety_compliance.compliance_rating.toLowerCase().replace(/\s+/g, "-")}`}>
                {report.safety_compliance.compliance_rating}
              </span>
            </div>
          </div>
          <div className="metric-grid" style={{ marginTop: 10 }}>
            <Metric value={`${report.coverage_comparison.novice_inspected_count}/${report.coverage_comparison.expert_inspected_count}`} label="Insulators vs expert" />
            <Metric value={`${report.safety_compliance.min_cable_distance_novice}m`} label="Novice min cable" tone={report.safety_compliance.min_cable_distance_novice < 2 ? "bad" : "good"} />
            <Metric value={report.safety_compliance.guardian_interventions_novice ?? 0} label="Guardian saves" />
          </div>
          <div className="section">
            <h4>Safety</h4>
            <p>{report.safety_compliance.commentary}</p>
            <h4>Coverage</h4>
            <p>{report.coverage_comparison.commentary}</p>
            <h4>Technique</h4>
            <p>{report.technique_and_stability.hover_discipline}</p>
            <h4>Expert rules applied</h4>
            <p>{report.technique_and_stability.knowledge_adherence}</p>
          </div>
          <div className="two-col">
            <div className="section">
              <h4>Strengths</h4>
              <ul className="bullets">{report.key_strengths.map((x) => <li key={x}>{x}</li>)}</ul>
            </div>
            <div className="section">
              <h4>To improve</h4>
              <ul className="bullets">{report.areas_for_improvement.map((x) => <li key={x}>{x}</li>)}</ul>
            </div>
          </div>
          <div className="verdict">{report.instructor_verdict}</div>
        </div>
      )}
    </div>
  );
}
