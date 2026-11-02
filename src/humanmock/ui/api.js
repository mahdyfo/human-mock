// Talks to the simulator. Paths are relative so the page also works behind a proxy.

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function readable(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const where = (item.loc || []).slice(1).join(" ");
        return where ? `${where}: ${item.msg}` : item.msg;
      })
      .join(" ");
  }
  return "The simulator returned an error.";
}

export async function api(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body === undefined ? {} : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "The simulator is not responding. Check that humanmock serve is still running.");
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, readable(data && data.detail));
  return data;
}
