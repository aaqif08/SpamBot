import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { ErrorBoundary } from "@/components/layout/ErrorBoundary";
import { EmptyState } from "@/components/ui";
import { AuthProvider, RequireAuth } from "@/hooks/useAuth";
import { HealthProvider } from "@/hooks/useHealth";
import { ThemeProvider } from "@/hooks/useTheme";
import { AppLayout } from "@/layouts/AppLayout";
import { AnalyzePage } from "@/pages/AnalyzePage";
import { ApiDocsPage } from "@/pages/ApiDocsPage";
import { ArchitecturePage } from "@/pages/ArchitecturePage";
import { BatchAnalysisPage } from "@/pages/BatchAnalysisPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { DatasetsPage } from "@/pages/DatasetsPage";
import { EvaluationPage } from "@/pages/EvaluationPage";
import { ExplainabilityPage } from "@/pages/ExplainabilityPage";
import { HistoryDetailPage, HistoryPage } from "@/pages/HistoryPage";
import { LoginPage } from "@/pages/LoginPage";
import { ModelsPage } from "@/pages/ModelsPage";
import { ResearchPage } from "@/pages/ResearchPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { TrainingPage } from "@/pages/TrainingPage";
import type { RoleName } from "@/types/api";

const ROUTES: { path: string; element: JSX.Element; roles?: RoleName[] }[] = [
  { path: "/analyze", element: <AnalyzePage />, roles: ["ADMIN", "ANALYST"] },
  { path: "/batch-analysis", element: <BatchAnalysisPage /> },
  { path: "/datasets", element: <DatasetsPage /> },
  { path: "/models", element: <ModelsPage /> },
  { path: "/training", element: <TrainingPage />, roles: ["ADMIN", "ANALYST"] },
  { path: "/evaluation", element: <EvaluationPage /> },
  { path: "/explainability", element: <ExplainabilityPage /> },
  { path: "/history", element: <HistoryPage /> },
  { path: "/history/:id", element: <HistoryDetailPage /> },
  { path: "/research", element: <ResearchPage /> },
  { path: "/architecture", element: <ArchitecturePage /> },
  { path: "/api-docs", element: <ApiDocsPage /> },
  { path: "/settings", element: <SettingsPage /> },
];

export function App() {
  return (
    <ThemeProvider>
      <HealthProvider>
        <BrowserRouter>
          <AuthProvider>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                element={
                  <RequireAuth>
                    <AppLayout />
                  </RequireAuth>
                }
              >
                <Route
                  index
                  element={
                    <ErrorBoundary>
                      <DashboardPage />
                    </ErrorBoundary>
                  }
                />
                {ROUTES.map((r) => (
                  <Route
                    key={r.path}
                    path={r.path}
                    element={
                      <RequireAuth roles={r.roles}>
                        <ErrorBoundary>{r.element}</ErrorBoundary>
                      </RequireAuth>
                    }
                  />
                ))}
                <Route path="/dashboard" element={<Navigate to="/" replace />} />
                <Route path="*" element={<EmptyState title="Page not found" description="The page you requested does not exist." />} />
              </Route>
            </Routes>
          </AuthProvider>
        </BrowserRouter>
      </HealthProvider>
    </ThemeProvider>
  );
}
