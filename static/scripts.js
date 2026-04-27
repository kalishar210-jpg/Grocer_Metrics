(function () {
  "use strict";

  const form = document.getElementById("predict-form");
  const submitBtn = document.getElementById("submit-btn");
  const btnLoader = document.getElementById("btn-loader");
  const errorBox = document.getElementById("error-box");
  const resultsEl = document.getElementById("results");

  const elProfit = document.getElementById("profit-result");
  const elDays = document.getElementById("days-to-sell");
  const elProb = document.getElementById("purchase-probability");
  const elRecos = document.getElementById("recommendations-list");

  if (!form) return;

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    hideError();
    setLoading(true);

    const formData = new FormData(form);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || "Unexpected server error.");
      }

      renderResults(data);
    } catch (err) {
      showError(err.message || "Could not connect to the prediction server.");
    } finally {
      setLoading(false);
    }
  });

  function renderResults(data) {
    const profit = parseFloat(data.sales_prediction);
    elProfit.textContent =
      isNaN(profit) ?
        data.sales_prediction
      : "£" +
        profit.toLocaleString("en-GB", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        });

    // Days to Sell
    const days = parseFloat(data.days_to_sell);
    elDays.textContent = isFinite(days) ? days.toLocaleString() : "∞";

    // Purchase Probability
    const prob = parseFloat(data.purchase_probability);
    elProb.textContent =
      isNaN(prob) ? data.purchase_probability : (prob * 100).toFixed(1) + "%";

    // Recommendations
    elRecos.innerHTML = "";
    const recos = data.recommendations;
    if (Array.isArray(recos) && recos.length > 0) {
      recos.forEach(function (item) {
        const pill = document.createElement("span");
        pill.className = "reco-pill";
        pill.textContent = item;
        elRecos.appendChild(pill);
      });
    } else {
      const empty = document.createElement("span");
      empty.className = "reco-empty";
      empty.textContent = "No recommendations found for this product.";
      elRecos.appendChild(empty);
    }

    // Show results section
    resultsEl.style.display = "grid";
    resultsEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ui helper
  function setLoading(loading) {
    submitBtn.disabled = loading;
    if (btnLoader) {
      btnLoader.classList.toggle("show", loading);
    }
  }

  function showError(msg) {
    if (!errorBox) return;
    errorBox.textContent = "⚠ " + msg;
    errorBox.style.display = "block";
  }

  function hideError() {
    if (!errorBox) return;
    errorBox.style.display = "none";
    errorBox.textContent = "";
  }
})();

