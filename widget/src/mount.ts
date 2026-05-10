/**
 * mount / unmount
 *
 * Vanilla-JS API for embedding the widget without a React app.
 *
 * Usage:
 *   VoiceRelayWidget.mount('#voice-widget', {
 *     url: 'wss://voice-relay.example.com/ws',
 *     apiKey: 'vr_live_...',
 *   });
 *
 *   // Later, to tear down:
 *   VoiceRelayWidget.unmount('#voice-widget');
 */

import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import type { Root } from 'react-dom/client';
import { VoiceRelay } from './VoiceRelay';
import type { VoiceRelayProps } from './types';

// Track mounted roots so we can unmount them later
const mountedRoots = new Map<Element, Root>();

/**
 * Resolve a selector or DOM element to an HTMLElement.
 */
function resolveContainer(target: string | Element): Element {
  if (typeof target === 'string') {
    const el = document.querySelector(target);
    if (!el) throw new Error(`[VoiceRelayWidget] No element found for selector: "${target}"`);
    return el;
  }
  return target;
}

/**
 * Mount the VoiceRelay widget into the given container.
 *
 * @param target - CSS selector string or DOM element
 * @param props  - VoiceRelayProps
 */
export function mount(
  target: string | Element,
  props: VoiceRelayProps,
): void {
  const container = resolveContainer(target);

  // Unmount any existing widget in this container first
  if (mountedRoots.has(container)) {
    unmount(container);
  }

  const root = createRoot(container);
  mountedRoots.set(container, root);
  root.render(createElement(VoiceRelay, props));
}

/**
 * Unmount the VoiceRelay widget from the given container.
 *
 * @param target - CSS selector string or DOM element
 */
export function unmount(target: string | Element): void {
  const container = resolveContainer(target);
  const root = mountedRoots.get(container);
  if (root) {
    root.unmount();
    mountedRoots.delete(container);
  }
}
