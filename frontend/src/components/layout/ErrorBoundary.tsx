import { Component, type ErrorInfo, type ReactNode } from "react";

import { ErrorState } from "@/components/ui";

interface Props {
  children: ReactNode;
}
interface State {
  error: Error | null;
}

/** Prevents a rendering error in one page from blanking the whole app. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Unhandled UI error", error, info);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="p-6">
          <ErrorState title="This view failed to render" message={this.state.error.message} onRetry={() => this.setState({ error: null })} />
        </div>
      );
    }
    return this.props.children;
  }
}
