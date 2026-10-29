(() => {
  const $ = id => document.getElementById(id);
  let selected = null;
  const endpoint = () => `/api/runs/${encodeURIComponent(selected)}`;
  async function api(url) { const res = await fetch(url); const data = await res.json(); if (!res.ok) throw Error(data.error || "기록을 읽을 수 없습니다."); return data; }
  function notice(error) { $("notice").textContent = String(error.message || error); }
  async function selectRun(name) {
    selected = name;
    const data = await api(endpoint());
    $("notice").textContent = `${data.config.model || "—"} · ${data.state.phase || "—"} · 채점 ${data.records.length}회`;
    $("report").src = `${endpoint()}/report`;
    $("report").hidden = false; $("downloads").hidden = false; $("inspector").hidden = false;
    for (const f of ["json", "csv", "html"]) $(f).href = `${endpoint()}/download?format=${f}`;
    $("trial").replaceChildren(); $("trace").replaceChildren();
    for (const r of data.records) { const option = document.createElement("option"); option.value = r.call; option.textContent = `#${r.call} ${r.problem_id} R${r.round} 시도 ${r.attempt} · ${r.feedback}`; $("trial").append(option); }
    for (const id of ["request", "response", "record"]) $(id).textContent = "";
  }
  async function refresh() {
    const data = await api("/api/runs"); $("runs").replaceChildren();
    for (const r of data.runs) { const b = document.createElement("button"); b.textContent = `${r.config.model} · ${r.name}`; b.addEventListener("click", () => selectRun(r.name).catch(notice)); $("runs").append(b); }
    if (!data.runs.length) $("runs").textContent = "아직 저장된 새 실험 기록이 없습니다.";
    if (selected) await selectRun(selected);
  }
  $("refresh").addEventListener("click", () => refresh().catch(notice));
  $("load-trial").addEventListener("click", async () => {
    if (!selected || !$("trial").value) return;
    try {
      const d = await api(`${endpoint()}/trials/${encodeURIComponent($("trial").value)}`);
      for (const id of ["request", "response", "record"]) $(id).textContent = JSON.stringify(d[id], null, 2);
      $("trace").replaceChildren();
      for (const [key, label] of [["current_trace", "현재 규칙"], ["previous_trace", "직전 규칙 · 현재 초기 배치·목표"]]) {
        const t = d[key]; if (!t) continue;
        const h = document.createElement("h3"); h.textContent = `${label}: ${t.reason}`; $("trace").append(h);
        const ol = document.createElement("ol");
        for (const step of t.steps) { const li = document.createElement("li"); li.textContent = `${step.index}: ${step.move ? `원판 ${step.disk ?? "?"} ${step.move.split("").join("→")}` : "초기 배치"} · ${step.message} · ${Object.entries(step.state).map(([k,v]) => `${k}=[${v}]`).join(" ")}`; ol.append(li); }
        $("trace").append(ol);
      }
      if (d.errors.length) notice(d.errors.join(" / "));
    } catch (e) { notice(e); }
  });
  refresh().catch(notice);
})();
