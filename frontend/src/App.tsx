import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";

import { ErrorBoundary } from "@/components/layout/ErrorBoundary";
import { EmptyState } from "@/components/ui";
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
import { ModelsPage } from "@/pages/ModelsPage";
import { ResearchPage } from "@/pages/ResearchPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { TrainingPage } from "@/pages/TrainingPage";

export function App() {
  return (
    <ThemeProvider>
      <HealthProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route
                index
                element={
                  <ErrorBoundary>
                    <DashboardPage />
                  </ErrorBoundary>
                }
              />
              {[
                ["/analyze", <AnalyzePage />],
                ["/batch-analysis", <BatchAnalysisPage />],
                ["/datasets", <DatasetsPage />],
                ["/models", <ModelsPage />],
                ["/training", <TrainingPage />],
                ["/evaluation", <EvaluationPage />],
                ["/explainability", <ExplainabilityPage />],
                ["/history", <HistoryPage />],
                ["/history/:id", <HistoryDetailPage />],
                ["/research", <ResearchPage />],
                ["/architecture", <ArchitecturePage />],
                ["/api-docs", <ApiDocsPage />],
                ["/settings", <SettingsPage />],
              ].map(([path, el]) => (
                <Route key={path as string} path={path as string} element={<ErrorBoundary>{el}</ErrorBoundary>} />
              ))}
              <Route path="/dashboard" element={<Navigate to="/" replace />} />
              <Route path="*" element={<EmptyState title="Page not found" description="The page you requested does not exist." />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </HealthProvider>
    </ThemeProvider>
  );
}
