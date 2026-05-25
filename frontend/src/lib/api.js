const API = "https://web-production-a6b67.up.railway.app";

export async function apiFetch(path, options = {}) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res;
}

export async function apiJson(path, options = {}) {
  const res = await apiFetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  return res.json();
}

export function getApiBase() {
  return API;
}