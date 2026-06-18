import { Outlet } from 'react-router-dom';

export default function Layout() {
  return (
    <div className="app-shell app-shell--console">
      <Outlet />
    </div>
  );
}
