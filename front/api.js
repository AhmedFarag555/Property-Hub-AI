const API_BASE = "http://127.0.0.1:8000";

async function apiRequest(endpoint, method = "GET", data = null, token = null, isForm = false) {

    const options = {
        method: method,
        headers: {}
    };

    // Token
    if (token) {
        options.headers["Authorization"] = `Bearer ${token}`;
    }

    // FormData
    if (isForm) {

        options.headers["Content-Type"] = "application/x-www-form-urlencoded";

        options.body = new URLSearchParams(data);

    } else if (data) {

        options.headers["Content-Type"] = "application/json";

        options.body = JSON.stringify(data);
    }

    const response = await fetch(`${API_BASE}${endpoint}`, options);

    if (!response.ok) {

        const errorData = await response.json();

        throw new Error(errorData.detail || "API Error");
    }

    return await response.json();
}