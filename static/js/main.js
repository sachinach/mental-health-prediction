/**
 * Mental Health Risk Prediction — site behaviour
 * Handles: dark mode toggle, scroll-reveal animation, prediction form
 * submission UX (spinner + validation), history search/pagination,
 * and lightweight toast notifications.
 */

(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    initThemeToggle();
    initScrollReveal();
    initPredictionForm();
    initHistoryTable();
    initDial();
  });

  // ---------------------------------------------------------------------
  // Dark mode
  // ---------------------------------------------------------------------
  function initThemeToggle() {
    const toggle = document.getElementById("themeToggle");
    if (!toggle) return;

    const stored = getStoredTheme();
    if (stored === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
      toggle.checked = true;
    }

    toggle.addEventListener("change", function () {
      if (toggle.checked) {
        document.documentElement.setAttribute("data-theme", "dark");
        setStoredTheme("dark");
      } else {
        document.documentElement.removeAttribute("data-theme");
        setStoredTheme("light");
      }
    });
  }

  // In-memory fallback since artifacts/sandboxed contexts may block
  // localStorage - safe to use directly here (this is a real Django
  // static file served to a normal browser), but we guard anyway.
  function getStoredTheme() {
    try {
      return window.localStorage.getItem("mhrp-theme");
    } catch (e) {
      return window.__mhrpTheme || null;
    }
  }

  function setStoredTheme(value) {
    try {
      window.localStorage.setItem("mhrp-theme", value);
    } catch (e) {
      window.__mhrpTheme = value;
    }
  }

  // ---------------------------------------------------------------------
  // Scroll reveal
  // ---------------------------------------------------------------------
  function initScrollReveal() {
    const targets = document.querySelectorAll(".reveal-on-scroll");
    if (!targets.length) return;

    if (!("IntersectionObserver" in window)) {
      targets.forEach((el) => el.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );

    targets.forEach((el) => observer.observe(el));
  }

  // ---------------------------------------------------------------------
  // Prediction form: client-side validation, loading spinner, reset
  // ---------------------------------------------------------------------
  function initPredictionForm() {
    const form = document.getElementById("predictionForm");
    if (!form) return;

    form.addEventListener("submit", function (event) {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
        form.classList.add("was-validated");
        showToast("Please fill in every field before submitting.", "error");
        return;
      }
      showSpinner("Analyzing your responses…");
    });

    const resetBtn = document.getElementById("resetFormBtn");
    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        form.reset();
        form.classList.remove("was-validated");
        showToast("Form cleared.", "info");
      });
    }
  }

  function showSpinner(message) {
    const overlay = document.createElement("div");
    overlay.className = "spinner-overlay";
    overlay.innerHTML =
      '<div class="spinner-card">' +
      '<div class="spinner-border" style="color:#6e9887" role="status"></div>' +
      '<p class="mt-3 mb-0 font-mono" style="font-size:0.85rem;color:#4b5866">' +
      (message || "Working…") +
      "</p></div>";
    document.body.appendChild(overlay);
  }

  // ---------------------------------------------------------------------
  // History page: client-side pagination on top of server-rendered rows
  // ---------------------------------------------------------------------
  function initHistoryTable() {
    const table = document.getElementById("historyTable");
    if (!table) return;

    const rows = Array.from(table.querySelectorAll("tbody tr"));
    const pageSize = 10;
    let currentPage = 1;
    const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));

    const pager = document.getElementById("historyPager");

    function renderPage(page) {
      currentPage = page;
      rows.forEach((row, idx) => {
        const start = (page - 1) * pageSize;
        const end = start + pageSize;
        row.style.display = idx >= start && idx < end ? "" : "none";
      });
      renderPager();
    }

    function renderPager() {
      if (!pager) return;
      pager.innerHTML = "";
      if (totalPages <= 1) return;

      for (let i = 1; i <= totalPages; i++) {
        const li = document.createElement("li");
        li.className = "page-item" + (i === currentPage ? " active" : "");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "page-link";
        btn.textContent = i;
        btn.addEventListener("click", () => renderPage(i));
        li.appendChild(btn);
        pager.appendChild(li);
      }
    }

    renderPage(1);
  }

  // ---------------------------------------------------------------------
  // Signature "clarity dial" on the homepage hero
  // ---------------------------------------------------------------------
  function initDial() {
    const dial = document.querySelector(".clarity-dial .dial-arc");
    if (!dial) return;
    const circumference = 2 * Math.PI * 80 * 0.75; // matches SVG arc length
    const targetValue = parseFloat(dial.getAttribute("data-value") || "94");
    dial.style.strokeDasharray = circumference;
    dial.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => {
      const offset = circumference * (1 - targetValue / 100);
      dial.style.strokeDashoffset = offset;
    });
  }

  // ---------------------------------------------------------------------
  // Toast notifications
  // ---------------------------------------------------------------------
  window.showToast = function (message, type) {
    let container = document.getElementById("toastContainer");
    if (!container) {
      container = document.createElement("div");
      container.id = "toastContainer";
      container.className = "toast-container position-fixed bottom-0 end-0 p-3";
      container.style.zIndex = 1090;
      document.body.appendChild(container);
    }

    const colors = { error: "#b3563f", info: "#24344d", success: "#4f7768" };
    const toast = document.createElement("div");
    toast.className = "toast align-items-center border-0 show mb-2";
    toast.style.background = colors[type] || colors.info;
    toast.style.color = "#f3f1ec";
    toast.style.borderRadius = "10px";
    toast.innerHTML =
      '<div class="d-flex"><div class="toast-body">' +
      message +
      '</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>';
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.transition = "opacity 0.3s ease";
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  };

  function showToast(message, type) {
    window.showToast(message, type);
  }
})();
