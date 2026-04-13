const AUTH_KEY = "hrh_current_user";
const UPLOADS_KEY = "hrh_uploads";
const BACKEND_URL = "http://localhost:5000";

function getCurrentUser() {
  try { return JSON.parse(localStorage.getItem(AUTH_KEY) || "null"); }
  catch (e) { return null; }
}
function setCurrentUser(email) { localStorage.setItem(AUTH_KEY, JSON.stringify({ email })); }
function clearCurrentUser() { localStorage.removeItem(AUTH_KEY); }
function getUploads() {
  try { return JSON.parse(localStorage.getItem(UPLOADS_KEY) || "[]"); }
  catch (e) { return []; }
}
function saveUploads(list) { localStorage.setItem(UPLOADS_KEY, JSON.stringify(list)); }
function formatTimestamp(ts) {
  return new Date(ts).toLocaleString([], { year:"numeric", month:"short", day:"numeric", hour:"2-digit", minute:"2-digit" });
}

const UNITS = {
  age: "years", gender: "",
  fasting_glucose: "mg/dL", HbA1c: "%", HOMA_IR: "", RBS: "mg/dL", PPG: "mg/dL",
  hemoglobin: "g/dL", RBC: "×10⁶/µL", hematocrit: "%", MCV: "fL", MCH: "pg",
  MCHC: "g/dL", RDW: "%", platelets: "×10³/µL",
  neutrophils_pct: "%", lymphocytes_pct: "%", monocytes_pct: "%",
  eosinophils_pct: "%", basophils_pct: "%",
  TSH: "mIU/L", Free_T3: "pg/mL", Free_T4: "ng/dL", anti_TPO: "IU/mL",
  creatinine: "mg/dL", urea_BUN: "mg/dL", eGFR: "mL/min/1.73m²", uric_acid: "mg/dL",
  ALT: "U/L", AST: "U/L", ALP: "U/L", GGT: "U/L",
  bilirubin_total: "mg/dL", bilirubin_direct: "mg/dL",
  albumin: "g/dL", total_protein: "g/dL",
  total_cholesterol: "mg/dL", LDL: "mg/dL", HDL: "mg/dL",
  VLDL: "mg/dL", triglycerides: "mg/dL", non_HDL: "mg/dL",
  WBC: "×10³/µL", CRP: "mg/L", ESR: "mm/hr", procalcitonin: "ng/mL",
  Vitamin_D: "ng/mL", B12: "pg/mL", folate: "ng/mL",
  iron: "µg/dL", ferritin: "ng/mL", TIBC: "µg/dL", transferrin_sat: "%", zinc: "µg/dL",
  sodium: "mEq/L", potassium: "mEq/L", chloride: "mEq/L", bicarbonate: "mEq/L",
  calcium: "mg/dL", magnesium: "mg/dL", phosphorus: "mg/dL", anion_gap: "mEq/L",
  cortisol: "µg/dL", testosterone: "ng/dL", estrogen: "pg/mL",
  FSH: "mIU/mL", LH: "mIU/mL", prolactin: "ng/mL", DHEA_S: "µg/dL",
  PT: "seconds", INR: "", aPTT: "seconds", fibrinogen: "mg/dL",
  D_dimer: "µg/mL", thrombin_time: "seconds",
};

const FEATURE_GROUPS = {
  "Metabolic / Diabetes": ["fasting_glucose","HbA1c","HOMA_IR","RBS","PPG"],
  "Hematological": ["hemoglobin","RBC","hematocrit","MCV","MCH","MCHC","RDW","platelets",
    "neutrophils_pct","lymphocytes_pct","monocytes_pct","eosinophils_pct","basophils_pct"],
  "Thyroid": ["TSH","Free_T3","Free_T4","anti_TPO"],
  "Renal": ["creatinine","urea_BUN","eGFR","uric_acid"],
  "Liver": ["ALT","AST","ALP","GGT","bilirubin_total","bilirubin_direct","albumin","total_protein"],
  "Lipid / CV": ["total_cholesterol","LDL","HDL","VLDL","triglycerides","non_HDL"],
  "Infection / Inflammation": ["WBC","CRP","ESR","procalcitonin"],
  "Nutritional": ["Vitamin_D","B12","folate","iron","ferritin","TIBC","transferrin_sat","zinc"],
  "Electrolyte": ["sodium","potassium","chloride","bicarbonate","calcium","magnesium","phosphorus","anion_gap"],
  "Hormonal / Endocrine": ["cortisol","testosterone","estrogen","FSH","LH","prolactin","DHEA_S"],
  "Clotting": ["PT","INR","aPTT","fibrinogen","D_dimer","thrombin_time"],
};

const CLASS_ICONS = {
  Normal:"✅", Diabetes:"🩸", Hematological:"🔬", Thyroid:"🦋",
  Renal:"🫘", Liver:"🫁", Lipid_CV:"❤️", Infection_Inflammation:"🦠",
  Nutritional:"💊", Electrolyte:"⚡", Hormonal_Endocrine:"🔄", Clotting:"🩹"
};

function formatOutput(result) {
  if (!result) return "<p>No output available.</p>";

  const prediction  = result.prediction || "Unknown";
  const confidence  = result.confidence != null ? (result.confidence * 100).toFixed(2) : null;
  const allProbs    = result.all_probabilities || {};
  const features    = result.extracted_features || {};
  const lowConf     = result.low_confidence_warning;
  const summary     = result.summary || null;

  const icon      = CLASS_ICONS[prediction] || "🔍";
  const confColor = confidence >= 80 ? "#16a34a" : confidence >= 50 ? "#d97706" : "#dc2626";

  let html = `
    <div class="result-banner">
      <div class="result-icon">${icon}</div>
      <div class="result-main">
        <div class="result-label">Predicted Condition</div>
        <div class="result-diagnosis">${prediction.replace(/_/g," ")}</div>
        ${confidence != null ? `<div class="result-confidence" style="color:${confColor}">Confidence: ${confidence}%</div>` : ""}
        ${lowConf ? `<div class="result-warning">⚠️ Low confidence — result may be uncertain</div>` : ""}
      </div>
    </div>`;

  if (summary) {
    const { conclusion = "", next_steps = [] } = summary;
    const stepsHtml = next_steps.map(s => `<li>${s}</li>`).join("");
    html += `
      <div class="summary-section">
        <h3>📋 Clinical Summary</h3>
        <p class="summary-conclusion">${conclusion}</p>
        ${stepsHtml ? `<h4>Recommended Next Steps</h4><ul class="summary-steps">${stepsHtml}</ul>` : ""}
      </div>`;
  }

  const sortedProbs = Object.entries(allProbs).sort((a,b) => b[1]-a[1]);
  html += `<div class="prob-section"><h3>All Class Probabilities</h3><div class="prob-bars">`;
  for (const [cls, prob] of sortedProbs) {
    const pct   = (prob * 100).toFixed(1);
    const isTop = cls === prediction;
    html += `
      <div class="prob-row">
        <span class="prob-label ${isTop?"prob-label-top":""}">${cls.replace(/_/g," ")}</span>
        <div class="prob-track"><div class="prob-fill ${isTop?"prob-fill-top":""}" style="width:${pct}%"></div></div>
        <span class="prob-pct">${pct}%</span>
      </div>`;
  }
  html += `</div></div>`;

  let hasAny = false;
  let featHtml = `<div class="feat-section"><h3>Extracted Lab Values</h3>`;
  for (const [group, keys] of Object.entries(FEATURE_GROUPS)) {
    const found = keys.filter(k => features[k] !== null && features[k] !== undefined);
    if (!found.length) continue;
    hasAny = true;
    featHtml += `<div class="feat-group"><div class="feat-group-title">${group}</div><table class="feat-table">`;
    for (const k of found) {
      const unit = UNITS[k] ? `<span class="feat-unit">${UNITS[k]}</span>` : "";
      featHtml += `<tr>
        <td>${k.replace(/_/g," ")}</td>
        <td><strong>${features[k]}</strong> ${unit}</td>
      </tr>`;
    }
    featHtml += `</table></div>`;
  }
  featHtml += `</div>`;
  if (hasAny) html += featHtml;

  return html;
}

function ensureLoggedIn() {
  const user = getCurrentUser();
  if (!user || !user.email) { window.location.href = "login.html"; return null; }
  return user;
}

function initLoginPage() {
  const form    = document.getElementById("loginForm");
  const guestBtn= document.getElementById("guestBtn");
  const error   = document.getElementById("error");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const email = document.getElementById("email").value.trim();
    error.textContent = "";
    if (!email) { error.textContent = "Please enter a valid email."; return; }
    setCurrentUser(email);
    window.location.href = "dashboard.html";
  });

  guestBtn.addEventListener("click", () => {
    setCurrentUser("guest@example.com");
    window.location.href = "dashboard.html";
  });
}

function createUploadItem(upload) {
  const wrapper = document.createElement("div");
  wrapper.className = "upload-item";
  wrapper.setAttribute("role","listitem");

  const title  = document.createElement("div");
  const nameEl = document.createElement("h3");
  nameEl.textContent = upload.name;
  const meta = document.createElement("small");
  meta.textContent = `Uploaded ${formatTimestamp(upload.createdAt)}`;
  title.appendChild(nameEl);
  title.appendChild(meta);

  const actions = document.createElement("div");
  actions.className = "upload-actions";

  const viewButton = document.createElement("button");
  viewButton.type = "button";
  viewButton.textContent = "View output";
  viewButton.addEventListener("click", () => selectUpload(upload.id));

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.addEventListener("click", () => deleteUpload(upload.id));

  actions.appendChild(viewButton);
  actions.appendChild(deleteButton);
  wrapper.appendChild(title);
  wrapper.appendChild(actions);
  return wrapper;
}

function renderUploads(selectedId) {
  const uploadsList = document.getElementById("uploadsList");
  const noUploads   = document.getElementById("noUploads");
  const uploads     = getUploads();
  uploadsList.innerHTML = "";
  if (!uploads.length) { noUploads.style.display = "block"; return; }
  noUploads.style.display = "none";
  uploads.slice().sort((a,b) => b.createdAt - a.createdAt).forEach((upload) => {
    const item = createUploadItem(upload);
    if (upload.id === selectedId) {
      item.style.borderColor = "rgba(37,99,235,0.68)";
      item.style.backgroundColor = "rgba(37,99,235,0.08)";
    }
    uploadsList.appendChild(item);
  });
}

function selectUpload(uploadId) {
  const uploads  = getUploads();
  const target   = uploads.find((u) => u.id === uploadId);
  if (!target) return;

  localStorage.setItem("hrh_selected", uploadId);

  const outputArea  = document.getElementById("outputArea");
  const outputIntro = document.getElementById("outputIntro");
  outputIntro.style.display = "none";

  if (target.output) {
    outputArea.innerHTML = formatOutput(target.output);
  } else if (target.error) {
    outputArea.innerHTML = `<p class="result-warning">❌ Error: ${target.error}</p>`;
  } else {
    outputArea.innerHTML = `<p class="muted">Processing...</p>`;
  }

  renderUploads(uploadId);
}

function deleteUpload(uploadId) {
  const uploads = getUploads().filter((u) => u.id !== uploadId);
  saveUploads(uploads);
  localStorage.removeItem("hrh_selected");
  const outputArea  = document.getElementById("outputArea");
  const outputIntro = document.getElementById("outputIntro");
  outputArea.innerHTML = "";
  outputIntro.style.display = "block";
  renderUploads();
}

function setUploadStatus(uploadId, status, data) {
  const uploads = getUploads();
  const idx     = uploads.findIndex((u) => u.id === uploadId);
  if (idx === -1) return;
  if (status === "success") { uploads[idx].output = data; uploads[idx].error = null; }
  else                      { uploads[idx].error  = data; uploads[idx].output = null; }
  saveUploads(uploads);
}

function initDashboardPage() {
  const user = ensureLoggedIn();
  if (!user) return;

  document.getElementById("profileEmail").textContent = user.email;
  document.getElementById("greeting").textContent = `Welcome, ${user.email.split("@")[0]}!`;

  document.getElementById("logoutBtn").addEventListener("click", () => {
    clearCurrentUser();
    window.location.href = "login.html";
  });

  document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById("reportFile");
    const file      = fileInput.files[0];
    if (!file) return;

    if (file.type !== "application/pdf") { alert("Please upload a PDF file."); return; }

    const submitBtn = document.querySelector("#uploadForm button[type='submit']");
    submitBtn.disabled = true;
    submitBtn.textContent = "Analyzing...";

    const newUpload = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      name: file.name, createdAt: Date.now(), output: null, error: null,
    };
    const uploads = getUploads();
    uploads.push(newUpload);
    saveUploads(uploads);
    fileInput.value = "";
    selectUpload(newUpload.id);

    try {
      const formData = new FormData();
      formData.append("pdf", file);

      const response = await fetch(`${BACKEND_URL}/predict`, { method:"POST", body:formData });
      if (!response.ok) {
        const err = await response.json().catch(() => ({ error:"Server error" }));
        throw new Error(err.error || `HTTP ${response.status}`);
      }
      const result = await response.json();
      setUploadStatus(newUpload.id, "success", result);
      selectUpload(newUpload.id);
    } catch (err) {
      setUploadStatus(newUpload.id, "error", err.message);
      selectUpload(newUpload.id);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Upload & Analyze";
    }
  });

  const persisted = localStorage.getItem("hrh_selected");
  renderUploads(persisted);
  if (persisted) selectUpload(persisted);
}

function init() {
  const path = window.location.pathname.split("/").pop();
  if (path === "login.html")    { initLoginPage();    return; }
  if (path === "dashboard.html"){ initDashboardPage(); return; }
}

document.addEventListener("DOMContentLoaded", init);