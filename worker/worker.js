export default {
  async fetch(request, env, ctx) {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, HEAD, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, X-Final-Url, X-Proxy-Manual-Redirect",
        },
      });
    }

    try {
      const url = new URL(request.url);
      let targetUrl = url.pathname.slice(1) + url.search;

      if (!targetUrl.startsWith("http")) {
        return new Response("Invalid URL. Usage: /https://example.com", { 
            status: 400,
            headers: { "Access-Control-Allow-Origin": "*" }
        });
      }

      const isManualMode = request.headers.get("X-Proxy-Manual-Redirect") === "true" || 
                           url.searchParams.get("__proxy_redirect_mode") === "manual";

      if (targetUrl.includes("__proxy_redirect_mode=")) {
          try {
              const targetUrlObj = new URL(targetUrl);
              targetUrlObj.searchParams.delete("__proxy_redirect_mode");
              targetUrl = targetUrlObj.toString();
          } catch (e) {
          }
      }

      const fetchOptions = {
        method: request.method,
        headers: {
          "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        },
        redirect: isManualMode ? "manual" : "follow"
      };

      const response = await fetch(targetUrl, fetchOptions);

      const headers = new Headers(response.headers);
      headers.set("Access-Control-Allow-Origin", "*");
      headers.set("Access-Control-Expose-Headers", "X-Final-Url, Location");

      if (isManualMode) {
        if (response.status >= 300 && response.status < 400) {
            const location = response.headers.get("Location");
            if (location) {
                headers.set("X-Final-Url", location);
            }
        } else {
            headers.set("X-Final-Url", targetUrl);
        }
        
        return new Response(response.body, {
            status: response.status,
            headers: headers
        });
      } else {
        headers.set("X-Final-Url", response.url || targetUrl);
        
        return new Response(response.body, {
            status: response.status,
            headers: headers
        });
      }

    } catch (error) {
      return new Response(`Error: ${error.message}`, {
        status: 500,
        headers: { "Access-Control-Allow-Origin": "*" }
      });
    }
  },
};
