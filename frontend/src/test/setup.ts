import "@testing-library/jest-dom/vitest";

// Recharts measures its container; jsdom has no layout, so provide a stub.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }).ResizeObserver = ResizeObserverStub;
