import { useEffect, useState } from "react";
import { API_URL } from "./config";
import { Ring } from "./Ring";
import { KnowledgeCoverage } from "./useSimSocket";

type SlotEntry = {
  rule: string;
  conditions?: string[];
  confidence: "once" | "confirmed";
  confirmations?: number;
  evidence?: Record<string, number>;
  induced?: boolean;
  answers?: Array<{ q: string; a: string; t?: number; phase?: string }>;
  teachback?: { confirmed: boolean };
};
type GridView = {
  coverage: KnowledgeCoverage;
  tasks: Array<{
    id: string;
    name: string;
    slots: Array<{ key: string; label: string; type: string; learn: string; entry: SlotEntry | null }>;
  }>;
};

const EVIDENCE_WORDS: Record<string, string> = {
  insulator_dist_median: "insulator distance",
  cable_dz_median: "height vs cable",
  cable_dist_median: "cable distance",
  duration_s: "hold time",
  speed_median: "speed",
  pylon_dist_min: "closest to tower",
  tree_dist_min: "closest to tree",
  altitude_median: "height",
  altitude_max: "max height",
};

/** The apprentice's competence matrix: learned (amber), confirmed in later flights (green), empty. */
export function KnowledgeViewer({ refreshKey, currentTask }: { refreshKey?: number; currentTask?: string | null }) {
  const [grid, setGrid] = useState<GridView | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [content, setContent] = useState<string>("");
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");
  const [saved, setSaved] = useState(false);

  const fetchGrid = () =>
    fetch(`${API_URL}/knowledge/competence`)
      .then((r) => r.json())
      .then(setGrid)
      .catch((e) => console.error("Failed to load the competence grid:", e));
  const fetchKnowledge = () =>
    fetch(`${API_URL}/knowledge`)
      .then((r) => r.json())
      .then((d) => {
        setContent(d.content || "");
        setEditText(d.content || "");
      })
      .catch((e) => console.error("Failed to load knowledge.md:", e));

  useEffect(() => {
    fetchGrid();
    if (!editing) fetchKnowledge(); // don't overwrite an edit in progress
  }, [refreshKey]);

  const resetGrid = async () => {
    if (!confirm("Empty the competence grid and the observed habits? Everything the apprentice learned is deleted.")) return;
    await fetch(`${API_URL}/knowledge/competence`, { method: "DELETE" }).catch(() => {});
    setOpen(null);
    fetchGrid();
    fetchKnowledge();
  };

  const forget = async (key: string) => {
    if (!confirm("Forget this rule? The apprentice and the tutor will no longer use it.")) return;
    const res = await fetch(`${API_URL}/knowledge/competence/${key}`, { method: "DELETE" }).catch(() => null);
    if (res?.ok) {
      setGrid(await res.json());
      fetchKnowledge();
    }
  };

  const save = async () => {
    const res = await fetch(`${API_URL}/knowledge`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: editText }),
    }).catch(() => null);
    if (res?.ok) {
      setContent(editText);
      setEditing(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    }
  };

  const slots = grid?.tasks.flatMap((t) => t.slots) ?? [];
  const slot = slots.find((s) => s.key === open);
  const induced = slots.filter((s) => s.entry?.induced).length;
  const cov = grid?.coverage;

  return (
    <div className="kb">
      <div className="panel kb-hero">
        <Ring size={84} stroke={8} value={cov?.filled ?? 0} total={cov?.total ?? 29} secondary={cov?.confirmed ?? 0}>
          <text x="42" y="40" textAnchor="middle" fill="#fff" fontFamily="Orbitron" fontSize="17" fontWeight="700">
            {cov ? Math.round((100 * cov.filled) / cov.total) : 0}%
          </text>
          <text x="42" y="55" textAnchor="middle" fill="#5f7591" fontFamily="Rajdhani" fontSize="9" letterSpacing="2">
            LEARNED
          </text>
        </Ring>
        <div style={{ flex: 1 }}>
          <div className="panel-head" style={{ marginBottom: 8 }}>
            <span className="eyebrow">Competence matrix</span>
            <button className="btn btn-sm btn-abort" onClick={resetGrid}>
              Reset
            </button>
          </div>
          <div className="kb-stats">
            <div className="stat amber">
              <b>{cov?.filled ?? 0}</b>
              <span>Rules / {cov?.total ?? 29}</span>
            </div>
            <div className="stat green">
              <b>{cov?.confirmed ?? 0}</b>
              <span>Confirmed</span>
            </div>
            <div className="stat violet">
              <b>{induced}</b>
              <span>From habits</span>
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="matrix">
          {grid?.tasks.map((t) => (
            <div key={t.id} className={`matrix-row ${t.id === currentTask ? "active" : ""}`}>
              <div className="matrix-task">{t.id === currentTask ? "▸ " : ""}{t.name}</div>
              <div className="matrix-cells">
                {t.slots.map((s) => (
                  <button
                    key={s.key}
                    className={`cell ${s.entry?.confidence ?? "empty"} ${open === s.key ? "open" : ""}`}
                    onClick={() => setOpen(open === s.key ? null : s.key)}
                    title={s.entry?.rule ?? s.learn}
                  >
                    {s.label}
                    {s.entry?.confidence === "confirmed" && <span className="x">×{s.entry.confirmations}</span>}
                    {s.entry?.induced && <span className="habit">◆</span>}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
        {slot && (
          <div className="cell-detail" style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--line)" }}>
            <h4>{slot.label}</h4>
            {slot.entry ? (
              <>
                <p>{slot.entry.rule}</p>
                {slot.entry.conditions?.map((c) => (
                  <p key={c} className="cond">↳ {c}</p>
                ))}
                {slot.entry.evidence && Object.keys(slot.entry.evidence).length > 0 && (
                  <p className="evid">
                    MEASURED ·{" "}
                    {Object.entries(slot.entry.evidence)
                      .map(([k, v]) => `${EVIDENCE_WORDS[k] ?? k} ${v}`)
                      .join(" · ")}
                  </p>
                )}
                {slot.entry.induced && <p className="evid">◆ Induced from the expert's repeated behaviour, then confirmed by the expert.</p>}
                {slot.entry.teachback?.confirmed && <p className="evid" style={{ color: "var(--green)" }}>✓ Confirmed by the expert in the teach-back.</p>}
                {(() => {
                  const said = [...(slot.entry.answers ?? [])].reverse().find((a) => a.a && !a.a.startsWith("("));
                  return said ? <p className="said">“{said.a}”</p> : null;
                })()}
                <div className="row-actions" style={{ marginTop: 6 }}>
                  <button className="btn btn-sm btn-abort" onClick={() => forget(slot.key)} title="Take this rule off the record">
                    Forget this rule
                  </button>
                </div>
              </>
            ) : (
              <p className="dim">Not learned yet: {slot.learn}</p>
            )}
          </div>
        )}
      </div>

      <details className="panel raw-kb">
        <summary>knowledge.md — the tutor's full knowledge base</summary>
        <div className="row-actions" style={{ marginTop: 10 }}>
          <button className="btn btn-sm" onClick={() => { fetchKnowledge(); fetchGrid(); }}>
            ↻ Refresh
          </button>
          {editing ? (
            <>
              <button className="btn btn-sm btn-primary" onClick={save}>
                Save
              </button>
              <button className="btn btn-sm" onClick={() => setEditing(false)}>
                Cancel
              </button>
            </>
          ) : (
            <button className="btn btn-sm" onClick={() => setEditing(true)}>
              Edit
            </button>
          )}
          {saved && <span className="save-ok">✓ Saved</span>}
        </div>
        {editing ? (
          <textarea className="kb-editor" value={editText} onChange={(e) => setEditText(e.target.value)} />
        ) : (
          <pre>{content || "Loading…"}</pre>
        )}
      </details>
    </div>
  );
}
