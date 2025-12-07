let abortController = null;
let logsAcumulados = [];
let timerInterval, startTime;

// Inicialização
document.addEventListener("DOMContentLoaded", () => {
  carregarHistorico();
  toggleStrategy();
});

// UI Helpers
function mostrarNome() {
  const f = document.getElementById("fileInput").files[0];
  if (f) document.getElementById("fileName").innerText = f.name;
}

function toggleStrategy() {
  const engine = document.getElementById("engineSelect").value;
  const stratBox = document.getElementById("strategyBox");
  const info = document.getElementById("ollamaInfo");

  if (engine === "ollama") {
    stratBox.style.display = "none";
    info.style.display = "block";
  } else {
    stratBox.style.display = "block";
    info.style.display = "none";
  }
}

// Timer Logic
function startTimer() {
  document.getElementById("timerDisplay").style.display = "block";
  startTime = Date.now();
  timerInterval = setInterval(() => {
    document.getElementById("timerDisplay").innerText = `⏱️ ${(
      (Date.now() - startTime) /
      1000
    ).toFixed(1)}s`;
  }, 100);
}
function stopTimer() {
  clearInterval(timerInterval);
}

// Logging
function addLog(msg) {
  const consoleDiv = document.getElementById("logsConsole");
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerText = `> ${msg}`;
  consoleDiv.appendChild(line);
  consoleDiv.scrollTop = consoleDiv.scrollHeight;
  logsAcumulados.push(msg);
}

// Processamento
function cancelarRequisicao() {
  if (abortController) {
    abortController.abort();
    abortController = null;
    addLog("!!! CANCELADO PELO USUÁRIO !!!");
    stopTimer();
    aguardarLiberacaoServidor();
  }
}

async function aguardarLiberacaoServidor() {
  const btn = document.getElementById("btnProcessar");
  const btnCancel = document.getElementById("btnCancelar");

  btnCancel.style.display = "none";
  btn.style.display = "block";
  btn.disabled = true;
  btn.innerText = "LIMPANDO SERVIDOR (AGUARDE)...";
  addLog("Aguardando servidor liberar recursos...");

  const checkInterval = setInterval(async () => {
    try {
      const res = await fetch("/status");
      const data = await res.json();
      if (data.ocupado === false) {
        clearInterval(checkInterval);
        addLog("Servidor liberado.");
        resetUI();
      }
    } catch (e) {
      console.log(e);
    }
  }, 1000);
}

function resetUI() {
  const btn = document.getElementById("btnProcessar");
  btn.style.display = "block";
  btn.disabled = false;
  btn.innerText = "INICIAR ANÁLISE AO VIVO";
  document.getElementById("btnCancelar").style.display = "none";
}

async function iniciarAnalise() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput.files[0]) {
    alert("Selecione um arquivo!");
    return;
  }

  const engine = document.getElementById("engineSelect").value;
  const model = document.getElementById("modelSelect").value;
  const nomeArquivo = fileInput.files[0].name;

  // Reset Visual
  document.getElementById("resultado").style.display = "none";
  document.getElementById("logsConsole").innerHTML = "";
  logsAcumulados = [];
  document.getElementById("btnProcessar").style.display = "none";
  document.getElementById("btnCancelar").style.display = "block";

  startTimer();
  abortController = new AbortController();

  const formData = new FormData();
  formData.append("arquivo", fileInput.files[0]);
  formData.append("modelo", model);
  formData.append(
    "palavras_proibidas",
    document.getElementById("badWordsInput").value
  );
  formData.append("motor", engine);
  formData.append(
    "estrategia",
    document.querySelector('input[name="strategy"]:checked').value
  );

  try {
    const response = await fetch("/analisar", {
      method: "POST",
      body: formData,
      signal: abortController.signal,
    });

    if (response.status === 503) {
      stopTimer();
      addLog("ERRO: Servidor ocupado.");
      resetUI();
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (line.startsWith("LOG:")) addLog(line.substring(4));
        else if (line.startsWith("RESULT:")) {
          const data = JSON.parse(line.substring(7));
          stopTimer();
          finalizarProcesso(data, nomeArquivo);
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      stopTimer();
      addLog("ERRO DE CONEXÃO: " + e);
      resetUI();
    }
  }
}

function finalizarProcesso(data, nomeArquivo) {
  resetUI();
  if (!data.sucesso) {
    addLog("ERRO NO BACKEND: " + data.erro);
    return;
  }

  document.getElementById("conteudoResultado").innerHTML =
    gerarHtmlResultado(data);
  document.getElementById("resultado").style.display = "block";

  salvarNoHistorico({
    id: Date.now(),
    dataHora: new Date().toLocaleString(),
    arquivo: nomeArquivo,
    modelo: data.metricas.modelo_usado,
    resultado: data,
  });
}

function aplicarDestaque(texto, palavras, frases) {
  let t = texto;
  if (frases && frases.length > 0) {
    frases.forEach((frase) => {
      const safeFrase = frase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const regex = new RegExp(`(${safeFrase})`, "gi");
      t = t.replace(regex, '<span class="highlight-context">$1</span>');
    });
  }
  if (palavras && palavras.length > 0) {
    palavras.forEach((p) => {
      const regex = new RegExp(`(${p})`, "gi");
      t = t.replace(regex, '<span class="highlight-bad">$1</span>');
    });
  }
  return t;
}

function gerarHtmlResultado(data) {
  const a = data.analise;
  const veredito = a.eh_ofensivo
    ? "🚫 CONTEÚDO OFENSIVO"
    : "✅ CONTEÚDO SEGURO";
  const cor = a.eh_ofensivo ? "unsafe" : "safe";
  const textoComDestaque = aplicarDestaque(
    data.transcricao,
    a.motivo_palavras,
    a.frases_criticas
  );

  let detalhes = "";
  if (a.motivo_palavras.length)
    detalhes += `<div>🛑 <strong>Palavras Proibidas:</strong> ${a.motivo_palavras.join(
      ", "
    )}</div>`;
  detalhes += `<div>🤖 <strong>Motivo da IA:</strong> ${a.motivo_ia}</div>`;

  return `
        <div class="veredito-card ${cor}">${veredito}</div>
        <div class="alert alert-warning">${detalhes}</div>
        <div class="row g-2 mb-3">
            <div class="col"><div class="metric-card"><span class="metric-value">${data.metricas.tempo_transcricao}</span><span class="metric-label">Transcrição</span></div></div>
            <div class="col"><div class="metric-card"><span class="metric-value">${data.metricas.tempo_analise}</span><span class="metric-label">Análise</span></div></div>
            <div class="col"><div class="metric-card"><span class="metric-value">${data.metricas.tempo_total}</span><span class="metric-label">Total</span></div></div>
            <div class="col"><div class="metric-card"><span class="metric-value">${a.score_ofensivo}%</span><span class="metric-label">% Ofensivo</span></div></div>
        </div>
        <label class="fw-bold">📝 Transcrição:</label>
        <div class="transcription-box">${textoComDestaque}</div>
    `;
}

// Histórico
function salvarNoHistorico(item) {
  let h = JSON.parse(localStorage.getItem("iaUltHistory") || "[]");
  h.unshift(item);
  if (h.length > 10) h.pop();
  localStorage.setItem("iaUltHistory", JSON.stringify(h));
  carregarHistorico();
}

function carregarHistorico() {
  const h = JSON.parse(localStorage.getItem("iaUltHistory") || "[]");
  const list = document.getElementById("historyList");
  list.innerHTML = "";

  if (h.length === 0) {
    list.innerHTML =
      '<li class="list-group-item text-center text-muted">Nenhuma análise salva.</li>';
    return;
  }

  h.forEach((item) => {
    const li = document.createElement("li");
    li.className =
      "list-group-item history-item d-flex justify-content-between align-items-center";

    const ehOfensivo = item.resultado.analise.eh_ofensivo;
    const badgeClass = ehOfensivo ? "bg-danger" : "bg-success";
    const badgeText = ehOfensivo ? "Ofensivo" : "Seguro";

    li.innerHTML = `
            <div onclick="abrirModalHistorico(${item.id})" class="flex-grow-1">
                <div class="fw-bold">🎵 ${item.arquivo}</div>
                <small class="text-muted">📅 ${item.dataHora} | 🧠 ${item.modelo}</small>
            </div>
            <div class="text-end ms-3">
                <span class="badge ${badgeClass}">${badgeText}</span>
                <div class="small text-muted">⏱️ ${item.resultado.metricas.tempo_total}</div>
            </div>
            <button class="btn btn-sm text-danger ms-3" onclick="deletarItem(${item.id})">🗑️</button>
        `;
    list.appendChild(li);
  });
}

function abrirModalHistorico(id) {
  const h = JSON.parse(localStorage.getItem("iaUltHistory") || "[]");
  const item = h.find((i) => i.id === id);
  if (item) {
    document.getElementById("modalBody").innerHTML = gerarHtmlResultado(
      item.resultado
    );
    const modal = new bootstrap.Modal(document.getElementById("historyModal"));
    modal.show();
  }
}

function deletarItem(id) {
  let h = JSON.parse(localStorage.getItem("iaUltHistory") || "[]");
  h = h.filter((item) => item.id !== id);
  localStorage.setItem("iaUltHistory", JSON.stringify(h));
  carregarHistorico();
}

function limparHistorico() {
  localStorage.removeItem("iaUltHistory");
  carregarHistorico();
}
