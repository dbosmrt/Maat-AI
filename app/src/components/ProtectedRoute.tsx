import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthProvider";

export default function ProtectedRoute() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="auth-loading" role="status" aria-label="Loading authentication">
        <svg className="spinner" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" fill="none" strokeDasharray="31.4 31.4" strokeLinecap="round" />
        </svg>
      </div>
    );
  }

  if (!user) {
    // Redirect to login with the intended destination
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
}