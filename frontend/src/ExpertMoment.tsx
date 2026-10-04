import { useEffect } from "react";
import { API_URL } from "./config";
import { ExpertMoment } from "./useSimSocket";

export const PHASE_LABEL: Record<string, string> = {
  live: "live question",
  note: "own note",
  debrief: "debrief",
  "teach-back": "teach-back correction",
};

/** The camera frame recorded closest to a moment of a flight. */
export function frameUrl(session?: string | null, t?: number | null): string | null {
  return session && t !== undefined && t !== null ? `${API_URL}/session/${session}/frame?t=${t}` : null;
}

export function clockOf(t?: number | null): string {
  if (t === undefined || t === null) return "--:--";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Shown over the novice's view when the tutor steps in: what the expert saw, decided and said there. */
export function ExpertMomentCard({ moment, onClose }: { moment: ExpertMoment; onClose: () => void }) {
  useEffect(() => {
    const id = setTimeout(onClose, 15000);
    return () => clearTimeout(id);
  }, [moment]);
  const url = moment.frame ? frameUrl(moment.session, moment.t) : null;
  return (
    <div className="moment-card">
      <div className="moment-head">
        <span className="eyebrow">Expert's moment</span>
        <span className="dim mono">
          {moment.phase ? PHASE_LABEL[moment.phase] ?? moment.phase : "taught"} · {clockOf(moment.t)}
        </span>
        <button className="moment-close" onClick={onClose} title="Close">
          ✕
        </button>
      </div>
      {url && <img src={url} alt="What the expert saw at that moment" />}
      <div className="moment-rule">
        <b>{moment.slot_name}</b>
        {moment.rule}
      </div>
      {moment.quote && (
        <blockquote>
          “{moment.quote}”<cite>— the expert</cite>
        </blockquote>
      )}
    </div>
  );
}
