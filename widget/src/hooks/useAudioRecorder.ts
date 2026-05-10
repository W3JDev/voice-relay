/**
 * useAudioRecorder
 *
 * AudioWorklet-based microphone capture hook.
 * Emits Float32 PCM frames at 16 kHz via an `onFrame` callback.
 * Also maintains a `waveformData` Float32Array for visualization.
 */

import { useCallback, useRef, useState } from 'react';
import type { UseAudioRecorderReturn } from '../types';

// ---------------------------------------------------------------------------
// AudioWorklet processor code (injected as Blob URL to avoid an extra file)
// ---------------------------------------------------------------------------
const WORKLET_CODE = /* js */ `
class AudioCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) this.port.postMessage(input.slice());
    return true;
  }
}
registerProcessor('vr-audio-capture', AudioCaptureProcessor);
`;

interface UseAudioRecorderOptions {
  onFrame: (pcmFloat32: Float32Array) => void;
}

export function useAudioRecorder(
  options: UseAudioRecorderOptions,
): UseAudioRecorderReturn {
  const { onFrame } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [waveformData, setWaveformData] = useState<Float32Array>(
    new Float32Array(128),
  );

  const audioCtxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const recordingRef = useRef(false); // sync flag used inside callbacks

  // -------------------------------------------------------------------------
  // Waveform animation loop
  // -------------------------------------------------------------------------
  const startWaveformLoop = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;

    const buf = new Float32Array(analyser.frequencyBinCount);

    function tick() {
      if (!recordingRef.current) return;
      analyser.getFloatTimeDomainData(buf);
      setWaveformData(buf.slice());
      animFrameRef.current = requestAnimationFrame(tick);
    }
    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  const stopWaveformLoop = useCallback(() => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setWaveformData(new Float32Array(128));
  }, []);

  // -------------------------------------------------------------------------
  // Initialise AudioWorklet (re-entrant safe)
  // -------------------------------------------------------------------------
  async function initWorklet(ctx: AudioContext): Promise<void> {
    const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    try {
      await ctx.audioWorklet.addModule(url);
    } catch (err) {
      // If the processor was already registered during this session we can
      // safely ignore the error. Any other error is re-thrown.
      const msg = (err as Error).message ?? '';
      if (!msg.includes('already been')) throw err;
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  const start = useCallback(async () => {
    if (recordingRef.current) return;

    try {
      // --- AudioContext ---
      if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
        audioCtxRef.current = new AudioContext({ sampleRate: 16000 });
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') await ctx.resume();

      // --- Microphone stream ---
      if (
        !streamRef.current ||
        streamRef.current.getTracks().every((t) => t.readyState === 'ended')
      ) {
        streamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      }

      // --- AudioWorklet ---
      await initWorklet(ctx);

      // --- Node graph ---
      sourceRef.current = ctx.createMediaStreamSource(streamRef.current);

      analyserRef.current = ctx.createAnalyser();
      analyserRef.current.fftSize = 256;
      analyserRef.current.smoothingTimeConstant = 0.7;
      sourceRef.current.connect(analyserRef.current);

      workletRef.current = new AudioWorkletNode(ctx, 'vr-audio-capture');
      workletRef.current.port.onmessage = (e: MessageEvent<Float32Array>) => {
        if (!recordingRef.current) return;
        onFrame(e.data);
      };
      sourceRef.current.connect(workletRef.current);
      // Must connect to destination (or a silent gain) to keep the graph alive
      workletRef.current.connect(ctx.destination);

      recordingRef.current = true;
      setIsRecording(true);
      startWaveformLoop();
    } catch (err) {
      console.error('[VoiceRelay] Recorder start error:', err);
      throw err;
    }
  }, [onFrame, startWaveformLoop]);

  const stop = useCallback(() => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setIsRecording(false);
    stopWaveformLoop();

    // Disconnect and dispose nodes
    try { workletRef.current?.disconnect(); } catch (_) { /* ignore */ }
    try { sourceRef.current?.disconnect(); } catch (_) { /* ignore */ }
    try { analyserRef.current?.disconnect(); } catch (_) { /* ignore */ }

    workletRef.current = null;
    sourceRef.current = null;
    analyserRef.current = null;

    // Close AudioContext to release resources (recreated on next start)
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
      audioCtxRef.current.close().catch(() => { /* ignore */ });
      audioCtxRef.current = null;
    }

    // Release the microphone track
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, [stopWaveformLoop]);

  return { isRecording, waveformData, start, stop };
}
