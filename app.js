// Simple client-side auth + uploads demo

const AUTH_KEY = "hrh_current_user";
const UPLOADS_KEY = "hrh_uploads";

function getCurrentUser() {
  try {
    return JSON.parse(localStorage.getItem(AUTH_KEY) || "null");
  } catch (e) {
    return null;
  }
}

function setCurrentUser(email) {
  localStorage.setItem(AUTH_KEY, JSON.stringify({ email }));
}

function clearCurrentUser() {
  localStorage.removeItem(AUTH_KEY);
}

function getUploads() {
  try {
    return JSON.parse(localStorage.getItem(UPLOADS_KEY) || "[]");
  } catch (e) {
    return [];
  }
}

function saveUploads(list) {
  localStorage.setItem(UPLOADS_KEY, JSON.stringify(list));
}

function formatTimestamp(ts) {
  const d = new Date(ts);
  return d.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fakeOutputFor(fileName) {
  // Simulate a result from a remote analysis service.
  return `Analysis result for “${fileName}”\n\n` +
    "This is a placeholder output. In a real system, the analysis would come from your backend.\n" +
    "• Summary: No obvious concerns detected.\n" +
    "• Recommendation: Keep a copy of this report and consult your healthcare provider if needed.";
}

function ensureLoggedIn() {
  const user = getCurrentUser();
  if (!user || !user.email) {
    window.location.href = "login.html";
    return null;
  }
  return user;
}

function initLoginPage() {
  const form = document.getElementById("loginForm");
  const error = document.getElementById("error");

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const emailInput = document.getElementById("email");
    const email = emailInput.value.trim();

    error.textContent = "";

    if (!email) {
      error.textContent = "Please enter a valid email.";
      return;
    }

    setCurrentUser(email);
    window.location.href = "dashboard.html";
  });
}

function createUploadItem(upload) {
  const wrapper = document.createElement("div");
  wrapper.className = "upload-item";
  wrapper.setAttribute("role", "listitem");

  const title = document.createElement("div");
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
  viewButton.addEventListener("click", () => {
    selectUpload(upload.id);
  });

  const deleteButton = document.createElement("button");
  deleteButton.type = "button";
  deleteButton.textContent = "Delete";
  deleteButton.addEventListener("click", () => {
    deleteUpload(upload.id);
  });

  actions.appendChild(viewButton);
  actions.appendChild(deleteButton);

  wrapper.appendChild(title);
  wrapper.appendChild(actions);
  return wrapper;
}

function renderUploads(selectedId) {
  const uploadsList = document.getElementById("uploadsList");
  const noUploads = document.getElementById("noUploads");
  const uploads = getUploads();

  uploadsList.innerHTML = "";

  if (!uploads.length) {
    noUploads.style.display = "block";
    return;
  }

  noUploads.style.display = "none";
  uploads
    .slice()
    .sort((a, b) => b.createdAt - a.createdAt)
    .forEach((upload) => {
      const item = createUploadItem(upload);
      if (upload.id === selectedId) {
        item.style.borderColor = "rgba(37, 99, 235, 0.68)";
        item.style.backgroundColor = "rgba(37, 99, 235, 0.08)";
      }
      uploadsList.appendChild(item);
    });
}

function selectUpload(uploadId) {
  const uploads = getUploads();
  const target = uploads.find((u) => u.id === uploadId);
  if (!target) {
    return;
  }

  // Persist selection so it survives refresh.
  localStorage.setItem("hrh_selected", uploadId);

  const outputArea = document.getElementById("outputArea");
  const outputIntro = document.getElementById("outputIntro");

  outputIntro.style.display = "none";
  outputArea.textContent = "Generating output...";

  // Simulate remote processing delay.
  setTimeout(() => {
    outputArea.textContent = target.output || fakeOutputFor(target.name);
  }, 400);

  renderUploads(uploadId);
}

function deleteUpload(uploadId) {
  const uploads = getUploads().filter((u) => u.id !== uploadId);
  saveUploads(uploads);
  localStorage.removeItem("hrh_selected");
  const outputArea = document.getElementById("outputArea");
  const outputIntro = document.getElementById("outputIntro");
  outputArea.textContent = "";
  outputIntro.style.display = "block";
  renderUploads();
}

function initDashboardPage() {
  const user = ensureLoggedIn();
  if (!user) return;

  const profileEmail = document.getElementById("profileEmail");
  const greeting = document.getElementById("greeting");
  const logoutBtn = document.getElementById("logoutBtn");
  const uploadForm = document.getElementById("uploadForm");
  const fileInput = document.getElementById("reportFile");

  profileEmail.textContent = user.email;
  greeting.textContent = `Welcome, ${user.email.split("@")[0]}!`;

  logoutBtn.addEventListener("click", () => {
    clearCurrentUser();
    window.location.href = "login.html";
  });

  uploadForm.addEventListener("submit", (event) => {
    event.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      return;
    }

    const uploads = getUploads();
    const newUpload = {
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      name: file.name,
      createdAt: Date.now(),
      output: null,
    };

    uploads.push(newUpload);
    saveUploads(uploads);

    fileInput.value = "";
    selectUpload(newUpload.id);
  });

  const persisted = localStorage.getItem("hrh_selected");
  renderUploads(persisted);

  if (persisted) {
    selectUpload(persisted);
  }
}

function init() {
  const path = window.location.pathname.split("/").pop();

  if (path === "login.html") {
    initLoginPage();
    return;
  }

  if (path === "dashboard.html") {
    initDashboardPage();
    return;
  }
}

document.addEventListener("DOMContentLoaded", init);
