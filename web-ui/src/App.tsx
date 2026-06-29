import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import ConsolePage from './pages/Console';
import { AppDialogProvider } from './components/AppDialog';

export default function App() {
  return (
    <AppDialogProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/console" replace />} />
          <Route path="console" element={<ConsolePage />} />
          <Route path="*" element={<Navigate to="/console" replace />} />
        </Route>
      </Routes>
    </AppDialogProvider>
  );
}
