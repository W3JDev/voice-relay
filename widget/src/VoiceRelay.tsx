/**
 * VoiceRelay – main React component.
 *
 * Renders the voice widget UI and delegates all WebSocket / audio logic to
 * the useVoiceRelay hook.
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { VoiceRelayProps } from './types';
import { useVoiceRelay } from './hooks/useVoiceRelay';
import './styles.css';

// ---------------------------------------------------------------------------
// Small sub-components
// ---------------------------------------------------------------------------

const MicIcon: React.FC = () => (
  <svg
    className="vr-mic-icon"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
  >
    <rect x="9" y="2" width="6" height="12" rx="3" />
    <path d="M5 10a7 7 0 0 0 14 0" />
    <line x1="12" y1="19" x2="12" y2="22" />
    <line x1="8" y1="22" x2="16" y2="22" />
  </svg>
);

const StopIcon: React.FC = () => (
  <svg
    className="vr-mic-icon"
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden
  >
    <rect x="5" y="5" width="14" height="14" rx="2" />
  </svg>
);

// ---------------------------------------------------------------------------
// Waveform canvas renderer
// ---------------------------------------------------------------------------

interface WaveformProps {
  data: Float32Array;
  active: boolean;
}

const Waveform: React.FC<WaveformProps> = ({ data, active }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const dataRef = useRef<Float32Array>(data);

  // Keep dataRef current without triggering re-renders
  useEffect(() => {
    dataRef.current = data;
  });

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    // Resize canvas if needed
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, w, h);

    const bars = 48;
    const gap = 2;
    const barWidth = Math.max(2, (w - gap * (bars - 1)) / bars);
    const waveData = dataRef.current;
    const step = Math.max(1, Math.floor(waveData.length / bars));

    for (let i = 0; i < bars; i++) {
      const idx = Math.min(i * step, waveData.length - 1);
      const amplitude = active ? Math.abs(waveData[idx] ?? 0) : 0;
      const scaled = Math.min(amplitude * 4, 1.0);
      const barH = active ? Math.max(2, scaled * (h - 8)) : 2;
      const x = i * (barWidth + gap);
      const y = (h - barH) / 2;

      const g = Math.round(107 + (194 - 107) * (1 - scaled));
      const b = Math.round(53 + (247 - 53) * (1 - scaled));
      const alpha = active ? 0.4 + scaled * 0.6 : 0.15;

      ctx.fillStyle = `rgba(255,${g},${b},${alpha})`;
      ctx.beginPath();
      if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(x, y, barWidth, barH, 2);
      } else {
        ctx.rect(x, y, barWidth, barH);
      }
      ctx.fill();
    }

    if (active) {
      rafRef.current = requestAnimationFrame(draw);
    }
  }, [active]);

  useEffect(() => {
    draw();
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [draw]);

  // Handle resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => {
      if (!active) draw();
    });
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [active, draw]);

  return (
    <canvas
      ref={canvasRef}
      className="vr-waveform-canvas"
      aria-hidden
    />
  );
};

// ---------------------------------------------------------------------------
// Minimal Markdown → JSX renderer (no external deps)
// ---------------------------------------------------------------------------

function renderMarkdown(text: string): React.ReactNode {
  // Split by code blocks first
  const parts = text.split(/(```[\s\S]*?```)/g);
  const nodes: React.ReactNode[] = [];

  parts.forEach((part, pi) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      const code = part.slice(3, -3).replace(/^\w+\n/, '');
      nodes.push(<pre key={pi}><code>{code.trim()}</code></pre>);
      return;
    }

    // Process inline
    const inline = part.split(/(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g);
    inline.forEach((seg, si) => {
      const key = `${pi}-${si}`;
      if (seg.startsWith('`') && seg.endsWith('`')) {
        nodes.push(<code key={key}>{seg.slice(1, -1)}</code>);
      } else if (seg.startsWith('**') && seg.endsWith('**')) {
        nodes.push(<strong key={key}>{seg.slice(2, -2)}</strong>);
      } else if (seg.startsWith('*') && seg.endsWith('*')) {
        nodes.push(<em key={key}>{seg.slice(1, -1)}</em>);
      } else if (/^\[.+\]\(.+\)$/.test(seg)) {
        const m = seg.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
        if (m) {
          nodes.push(
            <a key={key} href={m[2]} target="_blank" rel="noopener noreferrer">
              {m[1]}
            </a>,
          );
        } else {
          nodes.push(seg);
        }
      } else {
        // Replace newlines with <br>
        const lines = seg.split('\n');
        lines.forEach((line, li) => {
          if (li > 0) nodes.push(<br key={`${key}-br-${li}`} />);
          if (line) nodes.push(line);
        });
      }
    });
  });

  return <>{nodes}</>;
}

// ---------------------------------------------------------------------------
// TranscriptPanel
// ---------------------------------------------------------------------------

interface TranscriptPanelProps {
  entries: ReturnType<typeof useVoiceRelay>['transcript'];
}

const TranscriptPanel: React.FC<TranscriptPanelProps> = ({ entries }) => {
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const el = bodyRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [entries]);

  if (entries.length === 0) return null;

  return (
    <div className="vr-transcript" role="log" aria-live="polite" aria-label="Conversation">
      <div className="vr-transcript-body" ref={bodyRef}>
        {entries.map((entry) => (
          <div
            key={entry.id}
            className={`vr-msg ${entry.role}${entry.partial ? ' partial' : ''}`}
          >
            <span className="vr-msg-speaker">
              {entry.role === 'user' ? 'You' : 'AI'}
            </span>
            {entry.role === 'assistant'
              ? renderMarkdown(entry.text)
              : entry.text}
          </div>
        ))}
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main VoiceRelay component
// ---------------------------------------------------------------------------

export const VoiceRelay: React.FC<VoiceRelayProps> = ({
  url,
  apiKey,
  theme = 'dark',
  compact = false,
  position = 'inline',
  showTranscript = true,
  continuous: continuousProp = false,
  onSessionStart,
  onTranscript,
  onResponse,
  onError,
  className = '',
  style,
}) => {
  const [continuous, setContinuous] = useState(continuousProp);

  // Sync prop changes
  useEffect(() => {
    setContinuous(continuousProp);
  }, [continuousProp]);

  const {
    connectionState,
    isRecording,
    vadActive,
    sessionId,
    transcript,
    status,
    error,
    bargeInVisible,
    waveformData,
    startRecording,
    stopRecording,
    toggleRecording,
    clearTranscript,
  } = useVoiceRelay({
    url,
    apiKey,
    continuous,
    onSessionStart,
    onTranscript,
    onResponse,
    onError,
  });

  // ---- Press-to-talk state -------------------------------------------------
  const pttActiveRef = useRef(false);

  const handlePttStart = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if (continuous) return;
      pttActiveRef.current = true;
      startRecording();
    },
    [continuous, startRecording],
  );

  const handlePttEnd = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      e.preventDefault();
      if (continuous) return;
      if (pttActiveRef.current) {
        pttActiveRef.current = false;
        stopRecording();
      }
    },
    [continuous, stopRecording],
  );

  const handleClick = useCallback(() => {
    if (!continuous) return;
    toggleRecording();
  }, [continuous, toggleRecording]);

  // ---- Keyboard shortcut (Space) – only when widget is focused / inline ----
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.code !== 'Space' || e.repeat) return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      e.preventDefault();
      toggleRecording();
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [toggleRecording]);

  // ---- Connection state label ----------------------------------------------
  const connLabel =
    connectionState === 'connected'
      ? 'Connected'
      : connectionState === 'reconnecting'
      ? 'Reconnecting'
      : connectionState === 'connecting'
      ? 'Connecting'
      : 'Disconnected';

  // ---- Mic button label ----------------------------------------------------
  const micLabel = isRecording
    ? continuous
      ? 'Listening…'
      : 'Recording'
    : continuous
    ? 'Tap to Talk'
    : 'Hold to Talk';

  // ---- Root class list -----------------------------------------------------
  const rootClasses = [
    'vr-widget',
    `vr-theme-${theme}`,
    `vr-position-${position}`,
    compact ? 'vr-compact' : '',
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={rootClasses} style={style} data-testid="vr-widget">
      <div className="vr-panel">
        {/* ---- Header ---- */}
        <div className="vr-header">
          <div className="vr-header-left">
            <span className="vr-title">Voice</span>
            <div className="vr-conn-badge" aria-label={`Connection: ${connLabel}`}>
              <span className={`vr-conn-dot ${connectionState}`} aria-hidden />
              <span>{connLabel}</span>
            </div>
          </div>
          <button
            className="vr-icon-btn"
            onClick={clearTranscript}
            title="Clear conversation"
            aria-label="Clear conversation"
          >
            {/* Trash icon */}
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth={2} strokeLinecap="round"
              strokeLinejoin="round" aria-hidden>
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6l-1 14H6L5 6" />
              <path d="M10 11v6M14 11v6" />
              <path d="M9 6V4h6v2" />
            </svg>
          </button>
        </div>

        {/* ---- Voice controls ---- */}
        <div className="vr-voice-area">
          {/* Waveform */}
          <div className="vr-waveform-wrap" aria-hidden>
            <Waveform data={waveformData} active={isRecording} />
          </div>

          {/* Mic button */}
          <button
            className={`vr-mic-btn${isRecording ? ' listening' : ''}`}
            disabled={connectionState === 'disconnected'}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            aria-pressed={isRecording}
            /* PTT events */
            onMouseDown={continuous ? undefined : handlePttStart}
            onMouseUp={continuous ? undefined : handlePttEnd}
            onMouseLeave={continuous ? undefined : handlePttEnd}
            onTouchStart={continuous ? undefined : handlePttStart}
            onTouchEnd={continuous ? undefined : handlePttEnd}
            /* Toggle in continuous mode */
            onClick={continuous ? handleClick : undefined}
          >
            {isRecording ? <StopIcon /> : <MicIcon />}
            <span className="vr-mic-btn-label">{micLabel}</span>
            <span className={`vr-vad-dot${vadActive ? ' active' : ''}`} aria-hidden />
          </button>

          {/* Continuous toggle + barge-in */}
          <div className="vr-controls-row">
            <label className="vr-toggle-wrap">
              <span className="vr-toggle">
                <input
                  type="checkbox"
                  checked={continuous}
                  onChange={(e) => setContinuous(e.target.checked)}
                  aria-label="Continuous listening mode"
                />
                <span className="vr-toggle-slider" />
              </span>
              <span className="vr-toggle-label">Continuous</span>
            </label>

            <span
              className={`vr-barge-in${bargeInVisible ? ' visible' : ''}`}
              aria-live="polite"
              aria-atomic
            >
              ⚡ Interrupted
            </span>
          </div>

          {/* Status */}
          <p className={`vr-status${isRecording ? ' active' : ''}`} aria-live="polite">
            {status}
          </p>
        </div>

        {/* ---- Error bar ---- */}
        {error && (
          <div className="vr-error" role="alert">
            {error}
          </div>
        )}

        {/* ---- Transcript ---- */}
        {showTranscript && !compact && (
          <TranscriptPanel entries={transcript} />
        )}

        {/* ---- Footer (session ID) ---- */}
        {sessionId && !compact && (
          <div className="vr-footer">
            <span
              className="vr-session-id"
              title={sessionId}
              aria-label={`Session ID: ${sessionId}`}
            >
              {sessionId.slice(0, 8)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceRelay;
