import { useEffect, useState } from "react";
import { API_URL } from "./config";
import { PHASE_LABEL, clockOf, frameUrl } from "./ExpertMoment";

type Moment = { session: string; t: number; frame: boolean } | null;
type RuleView = {
  slot: string;
  label: string;
  type: string;
  guardrail: boolean;
  rule: string;
  conditions: string[];
  reason: string;
  confidence: "once" | "confirmed";
  confirmations: number;
  teachback: boolean;
  words: { text: string; question: string; phase: string } | null;
  moment: Moment;
};
type Step = {
  id: number;
  task: string;
  title: string;
  t_start: number;
  t_end: number;
  off_record?: boolean;
  moment?: Moment;
  decision?: string;
  events?: Array<{ type: string; t: number; label?: string }>;
  reason?: RuleView | null;
  rules?: RuleView[];
  guardrails?: RuleView[];
  judgment_call?: boolean;
  open?: string[];
};
type WorkMapData = {
  session_id: string;
  mode: string;
  title: string;
  duration_s: number;
  steps: Step[];
  guardrails: RuleView[];
  counts: { steps: number; judgment_calls: number; guardrails: number; rules_in_expert_words: number; open_gaps: number };
  debrief: {
    phase: string | null;
    questions: number;
    answered: number;
    teach_back: {
      speech: string;
      confirmed: boolean;
      round: number;
      history?: Array<{ speech: string; reply: string | null }>;
      corrections?: Array<{ slot_name?: string; said: string }>;
    } | null;
  };
};
type SessionMeta = { session_id: string; mode: string; started_at?: number };

function Words({ r, onMoment, label }: { r: RuleView; onMoment: (m: Moment, caption: string) => void; label?: string }) {
  if (!r.words) return null;
  const caption = `Taught at ${clockOf(r.moment?.t)} · ${PHASE_LABEL[r.words.phase] ?? r.words.phase}`;
  return (
    <blockquote className="wm-quote">
      {label && <span className="wm-quote-label">{label}</span>}“{r.words.text}”
      <cite>
        — the expert, {PHASE_LABEL[r.words.phase] ?? r.words.phase}
        {r.moment && (
          <button className="wm-moment" onClick={() => onMoment(r.moment, caption)}>
            ▶ {clockOf(r.moment.t)}
          </button>
        )}
      </cite>
    </blockquote>
  );
}

/** Module 2's output: the expert flight as a clickable timeline. Every step shows the screen moment,
 * the decision, the reason in the expert's words and the guardrails around it. */
export function WorkMapView({ sessionId, refreshKey }: { sessionId: string | null; refreshKey?: number }) {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [sid, setSid] = useState<string>("");
  const [data, setData] = useState<WorkMapData | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const [view, setView] = useState<{ moment: Moment; caption: string } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/sessions`)
      .then((r) => r.json())
      .then((d) => {
        const experts: SessionMeta[] = (d.sessions || []).filter((s: SessionMeta) => s.mode === "expert");
        setSessions(experts);
        const current = experts.find((s) => s.session_id === sessionId);
        setSid((prev) => (current ? current.session_id : prev || experts[0]?.session_id || ""));
      })
      .catch(() => {});
  }, [sessionId, refreshKey]);

  useEffect(() => {
    if (!sid) return;
    setLoading(true);
    fetch(`${API_URL}/workmap/${sid}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: WorkMapData | null) => {
        setData(d);
        const first = d?.steps.find((s) => s.judgment_call && !s.off_record) ?? d?.steps[0];
        setOpen(first?.id ?? null);
        setView(null);
      })
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [sid, refreshKey]);

  const step = data?.steps.find((s) => s.id === open) ?? null;
  const showMoment = (m: Moment, caption: string) => setView({ moment: m, caption });
  const shown = view ?? (step?.moment ? { moment: step.moment, caption: `Screen moment · ${clockOf(step.moment.t)} in this flight` } : null);
  const img = shown?.moment?.frame ? frameUrl(shown.moment.session, shown.moment.t) : null;
  const tb = data?.debrief.teach_back;
  const label = (x: SessionMeta) =>
    `${x.session_id}${x.started_at ? " · " + new Date(x.started_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""}`;

  return (
    <div className="panel workmap">
      <div className="panel-head">
        <span className="eyebrow">Work Map</span>
        <select className="select wm-select" value={sid} onChange={(e) => setSid(e.target.value)}>
          {sessions.length === 0 && <option value="">No expert flight yet</option>}
          {sessions.map((x) => (
            <option key={x.session_id} value={x.session_id}>
              {label(x)}
            </option>
          ))}
        </select>
      </div>
      {loading && <div className="loading-bar" />}
      {!data ? (
        <p className="summary-text dim">Fly as the expert, then answer the debrief: the Work Map is built from that flight.</p>
      ) : (
        <>
          <div className="wm-counts">
            <span className="chip">{data.counts.steps} steps</span>
            <span className="chip warn">⚖ {data.counts.judgment_calls} judgment calls</span>
            <span className="chip ok">◆ {data.counts.guardrails} guardrails</span>
            <span className="chip ai">“ ” {data.counts.rules_in_expert_words} rules in the expert's words</span>
            <span className={`chip ${tb?.confirmed ? "ok" : ""}`}>
              {tb?.confirmed ? "✓ Teach-back confirmed" : data.debrief.phase ? `Debrief: ${data.debrief.phase}` : "No debrief yet"}
            </span>
          </div>

          <div className="wm-timeline">
            {data.steps.map((s) => (
              <button
                key={s.id}
                className={`wm-step ${open === s.id ? "open" : ""} ${s.off_record ? "off" : ""} ${s.judgment_call ? "judgment" : ""}`}
                onClick={() => {
                  setOpen(s.id);
                  setView(null);
                }}
              >
                <span className="wm-time">{clockOf(s.t_start)}</span>
                <span className="wm-dot" />
                <span className="wm-title">
                  {s.id}. {s.title}
                </span>
                <span className="wm-badges">
                  {s.judgment_call && <i title="Judgment call">⚖</i>}
                  {!!s.guardrails?.length && <i className="g" title="Guardrails">◆{s.guardrails.length}</i>}
                  {!!s.rules?.some((r) => r.words) && <i className="q" title="Reason in the expert's words">“</i>}
                </span>
              </button>
            ))}
          </div>

          {step && !step.off_record && (
            <div className="wm-detail">
              {img ? (
                <figure className="wm-frame">
                  <img src={img} alt="Screen moment" />
                  <figcaption>{shown?.caption}</figcaption>
                </figure>
              ) : (
                <div className="wm-frame empty">No frame recorded at this moment</div>
              )}
              <div className="section">
                <h4>Decision</h4>
                <p>{step.decision}</p>
                <h4>Reason · in the expert's words</h4>
                {step.reason ? (
                  <>
                    <p className="wm-rule">{step.reason.rule}</p>
                    <Words r={step.reason} onMoment={showMoment} />
                  </>
                ) : (
                  <p className="dim">Not explained yet: the apprentice asks in the next debrief.</p>
                )}
                {!!step.guardrails?.length && (
                  <>
                    <h4>Guardrails</h4>
                    {step.guardrails.map((g) => (
                      <div key={g.slot} className="wm-guard">
                        <div>
                          <b>◆ {g.label}</b> {g.rule}
                          {g.teachback && <span className="wm-tb" title="Confirmed in the teach-back">✓</span>}
                        </div>
                        {g.conditions.map((c) => (
                          <div key={c} className="wm-cond">
                            ↳ {c}
                          </div>
                        ))}
                        <Words r={g} onMoment={showMoment} />
                      </div>
                    ))}
                  </>
                )}
                {!!step.rules?.filter((r) => !r.guardrail && r.slot !== step.reason?.slot).length && (
                  <>
                    <h4>Other rules</h4>
                    {step.rules
                      .filter((r) => !r.guardrail && r.slot !== step.reason?.slot)
                      .map((r) => (
                        <div key={r.slot} className="wm-guard soft">
                          <div>
                            <b>{r.label}</b> {r.rule}
                          </div>
                          <Words r={r} onMoment={showMoment} />
                        </div>
                      ))}
                  </>
                )}
                {!!step.open?.length && (
                  <>
                    <h4>Not taught yet</h4>
                    <div className="target-chips">
                      {step.open.map((o) => (
                        <span key={o} className="tchip">
                          {o}
                        </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
          {step?.off_record && <p className="summary-text dim">This part was flown off the record: nothing was seen or learned.</p>}

          {tb && (
            <div className={`teachback ${tb.confirmed ? "ok" : ""}`} style={{ marginTop: 12 }}>
              <div className="tb-head">
                Teach-back{tb.round > 1 ? ` · ${tb.round} rounds` : ""}
                {tb.confirmed ? <b> ✓ confirmed by the expert</b> : <b className="pending"> not confirmed yet</b>}
              </div>
              <p>{tb.history?.[0]?.speech ?? tb.speech}</p>
              {!!tb.corrections?.length && (
                <ul className="bullets">
                  {tb.corrections.map((c, i) => (
                    <li key={i}>
                      Corrected{c.slot_name ? ` (${c.slot_name})` : ""}: “{c.said}”
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
