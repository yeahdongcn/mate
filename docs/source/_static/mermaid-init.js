window.addEventListener("DOMContentLoaded", async () => {
  if (typeof mermaid === "undefined") {
    return;
  }

  mermaid.initialize({
    startOnLoad: true,
    securityLevel: "loose",
    theme: "neutral",
    themeVariables: {
      fontFamily: "system-ui, sans-serif",
      fontSize: "11px",
      primaryColor: "#f8fafc",
      primaryTextColor: "#111827",
      primaryBorderColor: "#2563eb",
      lineColor: "#475569",
      clusterBkg: "#ffffff",
      clusterBorder: "#cbd5e1",
    },
    flowchart: {
      curve: "linear",
      htmlLabels: true,
      nodeSpacing: 50,
      rankSpacing: 70,
      useMaxWidth: true,
    },
  });
});
