"use client";

import {
    createContext,
    useContext,
    useEffect,
    useState,
    useCallback,
    type ReactNode,
} from "react";
import {
    type AuthUser,
    type LoginRequest,
    type RegisterRequest,
    login as apiLogin,
    register as apiRegister,
    logout as apiLogout,
    getCurrentUser,
    getStoredToken,
    getStoredUser,
    isAuthenticated as checkAuth,
    handleOAuthTokenFromUrl,
} from "@/lib/auth-api";

interface AuthContextType {
    user: AuthUser | null;
    isLoading: boolean;
    isAuthenticated: boolean;
    error: string | null;
    login: (data: LoginRequest) => Promise<void>;
    register: (data: RegisterRequest) => Promise<void>;
    logout: () => void;
    refreshUser: () => Promise<void>;
    clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<AuthUser | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Check for existing auth on mount
    useEffect(() => {
        const initAuth = async () => {
            // First, check for OAuth token in URL (from OAuth callback redirect)
            try {
                const oauthUser = await handleOAuthTokenFromUrl();
                if (oauthUser) {
                    setUser(oauthUser);
                    setIsLoading(false);
                    return;
                }
            } catch {
                // OAuth token invalid, continue with normal flow
            }
            
            // Check for stored token
            const token = getStoredToken();
            if (token) {
                try {
                    const currentUser = await getCurrentUser();
                    setUser(currentUser);
                } catch {
                    // Token invalid, clear it
                    apiLogout();
                    setUser(null);
                }
            } else {
                // Try to get cached user
                const cachedUser = getStoredUser();
                if (cachedUser) {
                    setUser(cachedUser);
                }
            }
            setIsLoading(false);
        };
        initAuth();
    }, []);

    const login = useCallback(async (data: LoginRequest) => {
        setIsLoading(true);
        setError(null);
        try {
            const result = await apiLogin(data);
            setUser(result.user);
        } catch (err) {
            const message = err instanceof Error ? err.message : "Login failed";
            setError(message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const register = useCallback(async (data: RegisterRequest) => {
        setIsLoading(true);
        setError(null);
        try {
            const result = await apiRegister(data);
            setUser(result.user);
        } catch (err) {
            const message = err instanceof Error ? err.message : "Registration failed";
            setError(message);
            throw err;
        } finally {
            setIsLoading(false);
        }
    }, []);

    const logout = useCallback(() => {
        apiLogout();
        setUser(null);
        setError(null);
    }, []);

    const refreshUser = useCallback(async () => {
        if (!checkAuth()) return;
        try {
            const currentUser = await getCurrentUser();
            setUser(currentUser);
        } catch {
            // Token invalid
            apiLogout();
            setUser(null);
        }
    }, []);

    const clearError = useCallback(() => {
        setError(null);
    }, []);

    return (
        <AuthContext.Provider
            value={{
                user,
                isLoading,
                isAuthenticated: user !== null,
                error,
                login,
                register,
                logout,
                refreshUser,
                clearError,
            }}
        >
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth(): AuthContextType {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
